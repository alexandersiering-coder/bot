"""Persistente Speicherung für Erinnerungen & Notizen (Postgres via asyncpg).

Ohne DATABASE_URL bleibt das Feature einfach aus (REMINDERS_ENABLED=False),
genau wie die Bring-Anbindung ohne ihre Env-Vars.
"""

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import asyncpg

DATABASE_URL = os.getenv("DATABASE_URL")
REMINDERS_ENABLED = bool(DATABASE_URL)

TIMEZONE = os.getenv("TIMEZONE", "Europe/Berlin")
TZ = ZoneInfo(TIMEZONE)

_WEEKDAYS_DE = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]

_pool: asyncpg.Pool | None = None


async def init_db() -> None:
    global _pool
    if not REMINDERS_ENABLED:
        return
    _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reminders (
                id SERIAL PRIMARY KEY,
                chat_id BIGINT NOT NULL,
                text TEXT NOT NULL,
                due_at TIMESTAMPTZ NOT NULL,
                recurrence TEXT NOT NULL DEFAULT 'once',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id SERIAL PRIMARY KEY,
                chat_id BIGINT NOT NULL,
                text TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )


async def close_db() -> None:
    if _pool is not None:
        await _pool.close()


def now_local_label() -> str:
    """Aktuelles Datum/Uhrzeit als Kontext fürs LLM, damit es relative
    Zeitangaben ('morgen um 9', 'nächsten Montag') korrekt umrechnen kann."""
    now = datetime.now(TZ)
    weekday = _WEEKDAYS_DE[now.weekday()]
    return f"{weekday}, {now.strftime('%Y-%m-%d %H:%M')} ({TIMEZONE})"


def _to_utc(due_at: str) -> datetime:
    """Interpretiert ein vom LLM geliefertes ISO-Datetime ohne Zeitzone als
    lokale Zeit (TIMEZONE) und wandelt es für die Speicherung nach UTC um."""
    dt = datetime.fromisoformat(due_at)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt.astimezone(timezone.utc)


async def create_reminder(chat_id: int, text: str, due_at: str, recurrence: str = "once") -> str:
    if recurrence not in ("once", "daily", "weekly"):
        recurrence = "once"
    due_utc = _to_utc(due_at)
    async with _pool.acquire() as conn:
        row_id = await conn.fetchval(
            "INSERT INTO reminders (chat_id, text, due_at, recurrence) "
            "VALUES ($1, $2, $3, $4) RETURNING id",
            chat_id, text, due_utc, recurrence,
        )
    local = due_utc.astimezone(TZ)
    return f"Erinnerung #{row_id} gesetzt: '{text}' am {local.strftime('%Y-%m-%d %H:%M')} ({recurrence})."


async def list_reminders(chat_id: int) -> str:
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, text, due_at, recurrence FROM reminders "
            "WHERE chat_id = $1 ORDER BY due_at ASC",
            chat_id,
        )
    if not rows:
        return "Keine Erinnerungen vorhanden."
    lines = []
    for row in rows:
        local = row["due_at"].astimezone(TZ)
        suffix = f" (wiederholt: {row['recurrence']})" if row["recurrence"] != "once" else ""
        lines.append(f"#{row['id']}: {row['text']} — {local.strftime('%Y-%m-%d %H:%M')}{suffix}")
    return "\n".join(lines)


async def delete_reminder(chat_id: int, reminder_id: int) -> str:
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM reminders WHERE id = $1 AND chat_id = $2", reminder_id, chat_id
        )
    deleted = int(result.split()[-1])
    return "Erinnerung gelöscht." if deleted else "Keine Erinnerung mit dieser ID gefunden."


async def create_note(chat_id: int, text: str) -> str:
    async with _pool.acquire() as conn:
        row_id = await conn.fetchval(
            "INSERT INTO notes (chat_id, text) VALUES ($1, $2) RETURNING id", chat_id, text
        )
    return f"Notiz #{row_id} gespeichert."


async def list_notes(chat_id: int) -> str:
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, text FROM notes WHERE chat_id = $1 ORDER BY created_at ASC", chat_id
        )
    if not rows:
        return "Keine Notizen vorhanden."
    return "\n".join(f"#{row['id']}: {row['text']}" for row in rows)


async def delete_note(chat_id: int, note_id: int) -> str:
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM notes WHERE id = $1 AND chat_id = $2", note_id, chat_id
        )
    deleted = int(result.split()[-1])
    return "Notiz gelöscht." if deleted else "Keine Notiz mit dieser ID gefunden."


async def due_reminders() -> list[dict]:
    """Alle über alle Chats fälligen Erinnerungen (für den periodischen Check)."""
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, chat_id, text, due_at, recurrence FROM reminders WHERE due_at <= now()"
        )
    return [dict(row) for row in rows]


async def reschedule_or_delete(reminder_id: int, recurrence: str) -> None:
    async with _pool.acquire() as conn:
        if recurrence == "daily":
            await conn.execute(
                "UPDATE reminders SET due_at = due_at + INTERVAL '1 day' WHERE id = $1", reminder_id
            )
        elif recurrence == "weekly":
            await conn.execute(
                "UPDATE reminders SET due_at = due_at + INTERVAL '7 days' WHERE id = $1", reminder_id
            )
        else:
            await conn.execute("DELETE FROM reminders WHERE id = $1", reminder_id)
