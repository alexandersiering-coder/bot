"""Telegram-Chatbot mit LLM-Anbindung (Groq / Gemini / OpenAI-kompatibel)."""

import asyncio
import json
import logging
import os
import re
from collections import defaultdict, deque

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import AsyncOpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    AIORateLimiter,
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
API_KEY = os.environ["LLM_API_KEY"]
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Du bist ein hilfsbereiter Assistent in einem Telegram-Chat. "
    "Antworte knapp und in der Sprache des Nutzers.",
)
# Wie viele Nachrichten (User + Bot) pro Chat als Kontext mitgeschickt werden.
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "10"))
# Leer lassen = jeder darf schreiben. Sonst kommagetrennte Telegram-User-IDs.
ALLOWED_USERS = {
    int(uid) for uid in os.getenv("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if uid
}
# Name, auf den der Bot in Gruppen zusätzlich zu @mention/Reply reagiert.
NAME_TRIGGER = os.getenv("BOT_NAME_TRIGGER", "Kollege")
NAME_TRIGGER_RE = re.compile(rf"\b{re.escape(NAME_TRIGGER)}\b", re.IGNORECASE)

# Render setzt RENDER_EXTERNAL_URL automatisch; für andere Hosts WEBHOOK_URL
# manuell setzen. Ist keins von beiden gesetzt, läuft der Bot per Polling
# (praktisch für lokale Entwicklung).
WEBHOOK_URL = os.getenv("WEBHOOK_URL") or os.getenv("RENDER_EXTERNAL_URL")
PORT = int(os.getenv("PORT", "10000"))
# Telegram erlaubt für secret_token nur A-Z a-z 0-9 _ - (1-256 Zeichen);
# Renders generateValue liefert auch andere Zeichen, deshalb hier bereinigen.
_raw_webhook_secret = os.getenv("WEBHOOK_SECRET", "")
WEBHOOK_SECRET = re.sub(r"[^A-Za-z0-9_-]", "", _raw_webhook_secret)[:256] or None

# Bring!-Einkaufsliste über den separat deployten Bring-MCP-Server.
BRING_MCP_URL = os.getenv("BRING_MCP_URL")
BRING_MCP_TOKEN = os.getenv("BRING_MCP_TOKEN")
BRING_ENABLED = bool(BRING_MCP_URL and BRING_MCP_TOKEN)

if BRING_ENABLED:
    SYSTEM_PROMPT += (
        "\n\nDu hast über Functions Zugriff auf die Bring!-Einkaufsliste "
        "(list_shopping_lists, get_list_items, add_items, complete_items, "
        "remove_items). Nutze sie, wenn im Chat nach der Einkaufsliste "
        "gefragt wird oder etwas hinzugefügt, abgehakt oder entfernt "
        "werden soll. Frag nicht extra nach, ruf die Funktion direkt auf."
    )

BRING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_shopping_lists",
            "description": "Zeigt alle Bring!-Einkaufslisten mit Namen an.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_list_items",
            "description": "Liest offene und erledigte Artikel einer Bring!-Einkaufsliste.",
            "parameters": {
                "type": "object",
                "properties": {
                    "list_name": {
                        "type": "string",
                        "description": "Name der Liste. Ohne Angabe wird die erste Liste verwendet.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_items",
            "description": (
                "Setzt einen oder mehrere Artikel auf eine Bring!-Einkaufsliste. "
                "Für Mengen-/Sortenangaben 'Artikel: Angabe' nutzen, "
                "z. B. 'Milch: 2 Liter'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Artikel, die hinzugefügt werden sollen.",
                    },
                    "list_name": {
                        "type": "string",
                        "description": "Name der Liste. Ohne Angabe wird die erste Liste verwendet.",
                    },
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_items",
            "description": "Hakt Artikel auf einer Bring!-Einkaufsliste als gekauft ab.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Namen der Artikel, die abgehakt werden sollen.",
                    },
                    "list_name": {
                        "type": "string",
                        "description": "Name der Liste. Ohne Angabe wird die erste Liste verwendet.",
                    },
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_items",
            "description": (
                "Entfernt Artikel vollständig von einer Bring!-Einkaufsliste "
                "(anders als complete_items landen sie nicht bei erledigt, "
                "sondern verschwinden ganz)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Namen der Artikel, die entfernt werden sollen.",
                    },
                    "list_name": {
                        "type": "string",
                        "description": "Name der Liste. Ohne Angabe wird die erste Liste verwendet.",
                    },
                },
                "required": ["items"],
            },
        },
    },
]

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("bot")

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# chat_id -> letzte Nachrichten
history: dict[int, deque] = defaultdict(lambda: deque(maxlen=HISTORY_TURNS * 2))


def authorized(update: Update) -> bool:
    # In Gruppen/Supergruppen darf jedes Mitglied schreiben; die Allowlist
    # greift nur im privaten 1:1-Chat mit dem Bot.
    if update.effective_chat is not None and update.effective_chat.type in ("group", "supergroup"):
        return True
    if not ALLOWED_USERS:
        return True
    return update.effective_user is not None and update.effective_user.id in ALLOWED_USERS


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    await update.message.reply_text(
        "Hi! Schreib mir einfach etwas.\n\n"
        "/reset – Gesprächsverlauf löschen\n"
        "/model – aktuelles Modell anzeigen\n"
        "/help – diese Hilfe"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    history.pop(update.effective_chat.id, None)
    await update.message.reply_text("Verlauf gelöscht.")


async def model_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    await update.message.reply_text(f"Modell: {MODEL}\nEndpoint: {BASE_URL}")


async def call_bring_tool(name: str, arguments: dict) -> str:
    """Ruft ein Tool auf dem Bring-MCP-Server auf und gibt das Ergebnis als Text zurück."""
    async with streamablehttp_client(
        BRING_MCP_URL, headers={"Authorization": f"Bearer {BRING_MCP_TOKEN}"}
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            parts = [block.text for block in result.content if hasattr(block, "text")]
            text = "\n".join(parts) if parts else str(result.content)
            if result.isError:
                raise RuntimeError(text)
            return text


def directly_addressed(update: Update, bot_username: str) -> bool:
    """In Gruppen: nur bei @mention, Reply auf eine Bot-Nachricht oder Namensnennung reagieren."""
    message = update.message
    reply_to = message.reply_to_message
    if reply_to is not None and reply_to.from_user is not None and reply_to.from_user.username == bot_username:
        return True
    for entity in message.entities or []:
        if entity.type == "mention" and message.text[entity.offset : entity.offset + entity.length] == f"@{bot_username}":
            return True
    if NAME_TRIGGER_RE.search(message.text or ""):
        return True
    return False


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        await update.message.reply_text("Für diesen Bot bist du nicht freigeschaltet.")
        return

    is_group = update.effective_chat.type in ("group", "supergroup")
    if is_group and not directly_addressed(update, context.bot.username):
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text
    if is_group:
        # @mention aus dem Text entfernen, bevor er ans Modell geht.
        user_text = user_text.replace(f"@{context.bot.username}", "").strip()
    convo = history[chat_id]

    sender = update.effective_user.first_name if update.effective_user else "User"
    # In Gruppen kommen Nachrichten von verschiedenen Personen im selben Verlauf
    # an; der Name hilft dem Modell, Sprecher auseinanderzuhalten.
    prompt_text = f"{sender}: {user_text}" if is_group else user_text

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *convo,
                {"role": "user", "content": prompt_text}]
    llm_kwargs = {"tools": BRING_TOOLS, "tool_choice": "auto"} if BRING_ENABLED else {}

    typing = asyncio.create_task(keep_typing(context, chat_id))
    try:
        answer = None
        for _ in range(5):  # begrenzt, damit sich das Modell nicht endlos in Tool-Calls verrennt
            response = await client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0.7, max_tokens=1024, **llm_kwargs
            )
            msg = response.choices[0].message
            if not msg.tool_calls:
                answer = (msg.content or "").strip()
                break

            messages.append(msg.model_dump(exclude_none=True))
            for tool_call in msg.tool_calls:
                args = json.loads(tool_call.function.arguments or "{}")
                try:
                    result = await call_bring_tool(tool_call.function.name, args)
                except Exception as err:
                    log.warning("Bring-Tool-Call fehlgeschlagen: %s", err)
                    result = f"Fehler: {err}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        if answer is None:
            answer = "Ich komme gerade zu keinem Ergebnis, versuch's nochmal."
    except Exception:
        log.exception("LLM-Aufruf fehlgeschlagen")
        await update.message.reply_text("Da ist beim Modell etwas schiefgelaufen. Nochmal versuchen?")
        return
    finally:
        typing.cancel()

    convo.append({"role": "user", "content": prompt_text})
    convo.append({"role": "assistant", "content": answer})

    # Telegram-Limit: 4096 Zeichen pro Nachricht
    for i in range(0, len(answer), 4000):
        await update.message.reply_text(answer[i : i + 4000])


async def keep_typing(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    """Zeigt 'schreibt...' an, solange das Modell arbeitet."""
    try:
        while True:
            await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


def main() -> None:
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .rate_limiter(AIORateLimiter())
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("model", model_info))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    if WEBHOOK_URL:
        # Token im Pfad macht die URL selbst zum Geheimnis, damit niemand
        # ungefragt Updates an den Bot senden kann.
        webhook_path = f"/webhook/{TELEGRAM_TOKEN}"
        log.info(
            "Bot läuft im Webhook-Modus (Modell: %s) auf Port %s.", MODEL, PORT
        )
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=webhook_path,
            webhook_url=f"{WEBHOOK_URL.rstrip('/')}{webhook_path}",
            secret_token=WEBHOOK_SECRET,
            allowed_updates=Update.ALL_TYPES,
        )
    else:
        log.info("Bot läuft im Polling-Modus (Modell: %s). Beenden mit Strg+C.", MODEL)
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
