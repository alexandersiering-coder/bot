"""Persistente Speicherung für Erinnerungen & Notizen (Postgres via asyncpg).

Ohne DATABASE_URL bleibt das Feature einfach aus (REMINDERS_ENABLED=False),
genau wie die Bring-Anbindung ohne ihre Env-Vars.
"""

import os
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import asyncpg

DATABASE_URL = os.getenv("DATABASE_URL")
REMINDERS_ENABLED = bool(DATABASE_URL)

TIMEZONE = os.getenv("TIMEZONE", "Europe/Berlin")
TZ = ZoneInfo(TIMEZONE)

_WEEKDAYS_DE = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
# Reihenfolge entspricht datetime.weekday() (0=Montag ... 6=Sonntag).
_WEEKDAY_CODES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_WEEKDAY_LABELS = {"mon": "Mo", "tue": "Di", "wed": "Mi", "thu": "Do", "fri": "Fr", "sat": "Sa", "sun": "So"}

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
        # Nachträglich hinzugekommene Spalten (idempotent, auch für die
        # schon bestehende Tabelle in Neon).
        await conn.execute(
            "ALTER TABLE reminders ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'text'"
        )
        await conn.execute(
            "ALTER TABLE reminders ADD COLUMN IF NOT EXISTS recent_outputs TEXT[] NOT NULL DEFAULT '{}'"
        )
        # Forum-Gruppen: in welchem Thema wurde die Erinnerung angelegt?
        # NULL = normale Gruppe/Privatchat oder Alt-Eintrag -> Hauptchat.
        await conn.execute(
            "ALTER TABLE reminders ADD COLUMN IF NOT EXISTS thread_id BIGINT"
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
        # Wer will wie über Kalendertermine informiert werden?
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calendar_subs (
                chat_id BIGINT PRIMARY KEY,
                lead_minutes INT,
                briefing_time TEXT,
                last_briefing DATE
            )
            """
        )
        await conn.execute(
            "ALTER TABLE calendar_subs ADD COLUMN IF NOT EXISTS thread_id BIGINT"
        )
        # Verhindert, dass derselbe Termin mehrfach angekündigt wird.
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calendar_notified (
                chat_id BIGINT NOT NULL,
                event_uid TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (chat_id, event_uid)
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


def _normalize_recurrence(recurrence: str) -> str:
    """'once' / 'daily' / 'weekly:mon,wed,fri' (Kürzel s. _WEEKDAY_CODES).
    Alles andere/Ungültige fällt auf 'once' zurück."""
    recurrence = (recurrence or "once").strip().lower()
    if recurrence in ("once", "daily"):
        return recurrence
    if recurrence.startswith("weekly:"):
        given = {d.strip() for d in recurrence.removeprefix("weekly:").split(",")}
        ordered = [d for d in _WEEKDAY_CODES if d in given]
        if ordered:
            return "weekly:" + ",".join(ordered)
    return "once"


def _format_recurrence(recurrence: str) -> str:
    if recurrence == "once":
        return ""
    if recurrence == "daily":
        return " (wiederholt: täglich)"
    if recurrence.startswith("weekly:"):
        labels = ", ".join(_WEEKDAY_LABELS.get(d, d) for d in recurrence.removeprefix("weekly:").split(","))
        return f" (wiederholt: {labels})"
    return ""


async def create_reminder(
    chat_id: int, text: str, due_at: str, recurrence: str = "once", kind: str = "text",
    *, thread_id: int | None = None,
) -> str:
    recurrence = _normalize_recurrence(recurrence)
    kind = kind if kind in ("text", "vegan_recipe") else "text"
    due_utc = _to_utc(due_at)
    async with _pool.acquire() as conn:
        row_id = await conn.fetchval(
            "INSERT INTO reminders (chat_id, text, due_at, recurrence, kind, thread_id) "
            "VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
            chat_id, text, due_utc, recurrence, kind, thread_id,
        )
    local = due_utc.astimezone(TZ)
    return f"Erinnerung #{row_id} gesetzt: '{text}' am {local.strftime('%Y-%m-%d %H:%M')}{_format_recurrence(recurrence)}."


async def list_reminders(chat_id: int) -> str:
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, text, due_at, recurrence, kind FROM reminders "
            "WHERE chat_id = $1 ORDER BY due_at ASC",
            chat_id,
        )
    if not rows:
        return "Keine Erinnerungen vorhanden."
    lines = []
    for row in rows:
        local = row["due_at"].astimezone(TZ)
        kind_label = " [veganes Rezept]" if row["kind"] == "vegan_recipe" else ""
        lines.append(
            f"#{row['id']}: {row['text']}{kind_label} — "
            f"{local.strftime('%Y-%m-%d %H:%M')}{_format_recurrence(row['recurrence'])}"
        )
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
            "SELECT id, chat_id, thread_id, text, due_at, recurrence, kind, recent_outputs "
            "FROM reminders WHERE due_at <= now()"
        )
    return [dict(row) for row in rows]


async def record_output(reminder_id: int, output_id: str, keep: int = 8) -> None:
    """Merkt sich zuletzt verschickte Inhalte (z. B. Rezept-IDs) pro Erinnerung,
    damit sich wiederkehrende generierte Inhalte nicht wiederholen."""
    async with _pool.acquire() as conn:
        current = await conn.fetchval(
            "SELECT recent_outputs FROM reminders WHERE id = $1", reminder_id
        )
        updated = ((current or []) + [output_id])[-keep:]
        await conn.execute(
            "UPDATE reminders SET recent_outputs = $2 WHERE id = $1", reminder_id, updated
        )


async def set_calendar_notifications(
    chat_id: int, briefing_time: str | None = None, lead_minutes: int | None = None,
    *, thread_id: int | None = None,
) -> str:
    """Legt fest, wie dieser Chat über Kalendertermine informiert wird.
    Nicht übergebene Werte bleiben unverändert, 0/'off' schaltet einzeln ab.
    Das Thema (thread_id) folgt immer dem zuletzt eingestellten."""
    async with _pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO calendar_subs (chat_id) VALUES ($1) ON CONFLICT DO NOTHING", chat_id
        )
        await conn.execute(
            "UPDATE calendar_subs SET thread_id = $2 WHERE chat_id = $1", chat_id, thread_id
        )
        if briefing_time is not None:
            value = None if briefing_time.lower() in ("off", "aus", "") else briefing_time
            await conn.execute(
                "UPDATE calendar_subs SET briefing_time = $2, last_briefing = NULL WHERE chat_id = $1",
                chat_id, value,
            )
        if lead_minutes is not None:
            await conn.execute(
                "UPDATE calendar_subs SET lead_minutes = $2 WHERE chat_id = $1",
                chat_id, lead_minutes or None,
            )
        row = await conn.fetchrow(
            "SELECT briefing_time, lead_minutes FROM calendar_subs WHERE chat_id = $1", chat_id
        )

    parts = []
    parts.append(
        f"Tagesüberblick um {row['briefing_time']} Uhr" if row["briefing_time"]
        else "Tagesüberblick: aus"
    )
    parts.append(
        f"Vorab-Hinweis {row['lead_minutes']} Min. vor Terminen" if row["lead_minutes"]
        else "Vorab-Hinweis: aus"
    )
    return "Kalender-Benachrichtigungen: " + ", ".join(parts) + "."


async def calendar_subs() -> list[dict]:
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT chat_id, thread_id, lead_minutes, briefing_time, last_briefing "
            "FROM calendar_subs WHERE briefing_time IS NOT NULL OR lead_minutes IS NOT NULL"
        )
    return [dict(row) for row in rows]


async def mark_notified(chat_id: int, event_uid: str) -> bool:
    """True, wenn dieser Termin für diesen Chat noch nicht angekündigt war."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO calendar_notified (chat_id, event_uid) VALUES ($1, $2) "
            "ON CONFLICT DO NOTHING RETURNING chat_id",
            chat_id, event_uid,
        )
    return row is not None


async def mark_briefing_sent(chat_id: int, day: date) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE calendar_subs SET last_briefing = $2 WHERE chat_id = $1", chat_id, day
        )
        # Alte Dedupe-Einträge aufräumen, damit die Tabelle nicht unbegrenzt wächst.
        await conn.execute(
            "DELETE FROM calendar_notified WHERE created_at < now() - INTERVAL '7 days'"
        )


async def reschedule_or_delete(reminder_id: int, due_at: datetime, recurrence: str) -> None:
    async with _pool.acquire() as conn:
        if recurrence == "daily":
            await conn.execute(
                "UPDATE reminders SET due_at = due_at + INTERVAL '1 day' WHERE id = $1", reminder_id
            )
        elif recurrence.startswith("weekly:"):
            days = set(recurrence.removeprefix("weekly:").split(","))
            next_due = due_at
            for _ in range(7):
                next_due += timedelta(days=1)
                if _WEEKDAY_CODES[next_due.weekday()] in days:
                    break
            await conn.execute(
                "UPDATE reminders SET due_at = $2 WHERE id = $1", reminder_id, next_due
            )
        else:
            await conn.execute("DELETE FROM reminders WHERE id = $1", reminder_id)
