"""Telegram-Chatbot mit LLM-Anbindung (Groq / Gemini / OpenAI-kompatibel)."""

import asyncio
import base64
import io
import json
import logging
import os
import re
from collections import defaultdict, deque

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import AsyncOpenAI
from pypdf import PdfReader
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
# Eigenes Modell nur für Bilder: der Text-Default oben kann keine Bilder lesen.
# Läuft über denselben Endpoint/Key wie MODEL (BASE_URL), nur mit anderer
# Modell-ID. Falls dieses Modell bei deinem Provider nicht existiert, in der
# .env anpassen. qwen/qwen3.6-27b ist ein Reasoning-Modell (denkt in einem
# <think>-Block, bevor die Antwort kommt) — reasoning_format=hidden filtert
# das bei Groq direkt raus.
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.6-27b")
# Reasoning-Denkblock bei Groq-Reasoning-Modellen ausblenden. Bei anderen
# Providern (Gemini/OpenAI) kennt die API dieses Feld nicht -> weglassen.
VISION_EXTRA_BODY = {"reasoning_format": "hidden"} if "groq.com" in BASE_URL else None
# Max. Zeichen aus einer PDF, die ans Modell gehen (Kostenbremse bei langen PDFs).
PDF_MAX_CHARS = int(os.getenv("PDF_MAX_CHARS", "15000"))
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


def _format_error(err: BaseException) -> str:
    """Klartext auch für ExceptionGroups (z. B. 'unhandled errors in a TaskGroup')."""
    parts = [f"{type(err).__name__}: {err}"]
    for sub in getattr(err, "exceptions", []):
        parts.append(_format_error(sub))
    return " | caused by | ".join(parts)


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


def _trigger_source(message) -> tuple[str, list]:
    """Text+Entities einer Nachricht, egal ob normaler Text oder Bild-/PDF-Caption."""
    if message.text is not None:
        return message.text, message.entities or []
    return message.caption or "", message.caption_entities or []


def directly_addressed(update: Update, bot_username: str) -> bool:
    """In Gruppen: nur bei @mention, Reply auf eine Bot-Nachricht oder Namensnennung reagieren."""
    message = update.message
    reply_to = message.reply_to_message
    if reply_to is not None and reply_to.from_user is not None and reply_to.from_user.username == bot_username:
        return True
    text, entities = _trigger_source(message)
    for entity in entities:
        if entity.type == "mention" and text[entity.offset : entity.offset + entity.length] == f"@{bot_username}":
            return True
    if NAME_TRIGGER_RE.search(text):
        return True
    return False


async def ask_llm(chat_id: int, content, *, model: str = MODEL, use_tools: bool = True,
                   extra_body: dict | None = None) -> str:
    """Schickt `content` (String oder Multimodal-Content-Liste) ans Modell.

    Führt bei Bedarf die Bring-Tool-Call-Runde aus und gibt die finale
    Textantwort zurück. Aktualisiert NICHT den persistenten Verlauf — das
    macht der Aufrufer, weil er entscheidet, was davon als Kontext für
    künftige Turns gespeichert werden soll (z. B. keine rohen Bild-Bytes).
    """
    convo = history[chat_id]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *convo,
                {"role": "user", "content": content}]
    llm_kwargs = {"tools": BRING_TOOLS, "tool_choice": "auto"} if (use_tools and BRING_ENABLED) else {}
    if extra_body:
        llm_kwargs["extra_body"] = extra_body

    answer = None
    max_tokens = 2048 if extra_body else 1024  # Reasoning-Modelle brauchen Puffer fürs Denken
    for _ in range(5):  # begrenzt, damit sich das Modell nicht endlos in Tool-Calls verrennt
        response = await client.chat.completions.create(
            model=model, messages=messages, temperature=0.7, max_tokens=max_tokens, **llm_kwargs
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
                log.warning("Bring-Tool-Call fehlgeschlagen: %s", _format_error(err))
                result = f"Fehler: {err}"
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return answer if answer is not None else "Ich komme gerade zu keinem Ergebnis, versuch's nochmal."


async def reply_with_llm(update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id: int,
                          content, history_text: str, *, model: str = MODEL, use_tools: bool = True,
                          extra_body: dict | None = None) -> None:
    """Ruft ask_llm auf, zeigt währenddessen 'schreibt...' und schickt die Antwort.

    `history_text` ist die kompakte Text-Repräsentation, die im Verlauf
    landet (statt z. B. roher Bild-Daten).
    """
    typing = asyncio.create_task(keep_typing(context, chat_id))
    try:
        answer = await ask_llm(chat_id, content, model=model, use_tools=use_tools, extra_body=extra_body)
    except Exception:
        log.exception("LLM-Aufruf fehlgeschlagen")
        await update.message.reply_text("Da ist beim Modell etwas schiefgelaufen. Nochmal versuchen?")
        return
    finally:
        typing.cancel()

    convo = history[chat_id]
    convo.append({"role": "user", "content": history_text})
    convo.append({"role": "assistant", "content": answer})

    # Telegram-Limit: 4096 Zeichen pro Nachricht
    for i in range(0, len(answer), 4000):
        await update.message.reply_text(answer[i : i + 4000])


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

    sender = update.effective_user.first_name if update.effective_user else "User"
    # In Gruppen kommen Nachrichten von verschiedenen Personen im selben Verlauf
    # an; der Name hilft dem Modell, Sprecher auseinanderzuhalten.
    prompt_text = f"{sender}: {user_text}" if is_group else user_text

    await reply_with_llm(update, context, chat_id, prompt_text, prompt_text)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        await update.message.reply_text("Für diesen Bot bist du nicht freigeschaltet.")
        return

    is_group = update.effective_chat.type in ("group", "supergroup")
    if is_group and not directly_addressed(update, context.bot.username):
        return

    chat_id = update.effective_chat.id
    caption = (update.message.caption or "Was ist auf dem Bild zu sehen?").strip()
    if is_group and context.bot.username:
        caption = caption.replace(f"@{context.bot.username}", "").strip()

    photo = update.message.photo[-1]  # letztes = höchste Auflösung
    if photo.file_size and photo.file_size > 4 * 1024 * 1024:
        await update.message.reply_text("Das Bild ist zu groß (>4 MB), das schafft das Modell nicht.")
        return

    file = await context.bot.get_file(photo.file_id)
    raw = await file.download_as_bytearray()
    b64 = base64.b64encode(bytes(raw)).decode()

    content = [
        {"type": "text", "text": caption},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]
    # Roh-Bilddaten nicht im Verlauf speichern, nur einen Text-Platzhalter.
    history_text = f"[Bild gesendet] {caption}"

    await reply_with_llm(
        update, context, chat_id, content, history_text,
        model=VISION_MODEL, use_tools=False, extra_body=VISION_EXTRA_BODY,
    )


async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        await update.message.reply_text("Für diesen Bot bist du nicht freigeschaltet.")
        return

    is_group = update.effective_chat.type in ("group", "supergroup")
    if is_group and not directly_addressed(update, context.bot.username):
        return

    chat_id = update.effective_chat.id
    doc = update.message.document
    caption = (update.message.caption or "").strip()
    if is_group and context.bot.username:
        caption = caption.replace(f"@{context.bot.username}", "").strip()

    file = await context.bot.get_file(doc.file_id)
    raw = await file.download_as_bytearray()

    try:
        reader = PdfReader(io.BytesIO(bytes(raw)))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        log.exception("PDF konnte nicht gelesen werden")
        await update.message.reply_text("Konnte die PDF nicht lesen — ist sie beschädigt oder passwortgeschützt?")
        return

    if not text:
        await update.message.reply_text(
            "Aus der PDF ließ sich kein Text extrahieren — vermutlich ein Scan/Bild ohne Textebene."
        )
        return

    truncated = len(text) > PDF_MAX_CHARS
    text = text[:PDF_MAX_CHARS]

    prompt_text = f"[PDF: {doc.file_name}]"
    if caption:
        prompt_text += f"\n{caption}"
    prompt_text += f"\n\n{text}"
    if truncated:
        prompt_text += "\n\n[Text wurde gekürzt, PDF ist länger.]"

    history_text = f"[PDF gesendet: {doc.file_name}] {caption}".strip()

    await reply_with_llm(update, context, chat_id, prompt_text, history_text)


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
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))

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
