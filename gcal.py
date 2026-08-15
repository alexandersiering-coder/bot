"""Google-Kalender (lesen & schreiben) über ein Service-Account.

Bewusst NICHT `calendar.py` genannt — das würde das gleichnamige Modul der
Standardbibliothek überdecken, das andere Pakete importieren.

Ohne GOOGLE_SERVICE_ACCOUNT_JSON + GOOGLE_CALENDAR_ID bleibt das Feature aus,
genau wie Bring/Reminders/Rezepte ohne ihre jeweiligen Env-Vars.
"""

import asyncio
import base64
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

_RAW_CREDENTIALS = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID")
CALENDAR_ENABLED = bool(_RAW_CREDENTIALS and CALENDAR_ID)

TIMEZONE = os.getenv("TIMEZONE", "Europe/Berlin")
TZ = ZoneInfo(TIMEZONE)

_SCOPES = ["https://www.googleapis.com/auth/calendar"]
_WEEKDAYS_SHORT = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

_service = None
# googleapiclient-Service-Objekte sind nicht thread-safe, asyncio.to_thread
# kann aber wechselnde Threads verwenden -> Zugriffe serialisieren. Bei den
# paar Aufrufen pro Minute hier völlig unproblematisch.
_service_lock = threading.Lock()


def _load_credentials():
    from google.oauth2 import service_account

    raw = _RAW_CREDENTIALS.strip()
    if not raw.startswith("{"):  # erlaubt auch base64-kodiertes JSON
        raw = base64.b64decode(raw).decode()
    return service_account.Credentials.from_service_account_info(
        json.loads(raw), scopes=_SCOPES
    )


def _service_call(fn):
    global _service
    with _service_lock:
        if _service is None:
            from googleapiclient.discovery import build

            _service = build(
                "calendar", "v3", credentials=_load_credentials(), cache_discovery=False
            )
        return fn(_service)


def _parse_bound(value: str, *, is_end: bool = False) -> datetime:
    """ISO-Datum oder -Datetime aus dem LLM in eine zeitzonenbehaftete Grenze.
    Ein reines Datum als Ende schließt den ganzen Tag mit ein."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    if is_end and len(value) == 10:
        dt += timedelta(days=1)
    return dt


def event_start(event: dict) -> datetime:
    start = event.get("start", {})
    if "dateTime" in start:
        return datetime.fromisoformat(start["dateTime"]).astimezone(TZ)
    return datetime.fromisoformat(start["date"]).replace(tzinfo=TZ)


def is_all_day(event: dict) -> bool:
    return "date" in event.get("start", {})


def format_event(event: dict, *, with_id: bool = False) -> str:
    start = event_start(event)
    when = f"{_WEEKDAYS_SHORT[start.weekday()]}, {start.strftime('%d.%m.')}"
    when += " (ganztägig)" if is_all_day(event) else f" {start.strftime('%H:%M')}"

    line = f"{when} — {event.get('summary') or '(ohne Titel)'}"
    if event.get("location"):
        line += f" @ {event['location']}"
    if with_id:
        line += f"  [id: {event['id']}]"
    return line


def _list_sync(time_min: datetime, time_max: datetime, max_results: int) -> list[dict]:
    return _service_call(
        lambda s: s.events()
        .list(
            calendarId=CALENDAR_ID,
            timeMin=time_min.astimezone(timezone.utc).isoformat(),
            timeMax=time_max.astimezone(timezone.utc).isoformat(),
            singleEvents=True,  # Serientermine als Einzeltermine auflösen
            orderBy="startTime",
            maxResults=max_results,
        )
        .execute()
    ).get("items", [])


async def list_events(
    start: str | None = None, end: str | None = None, max_results: int = 25
) -> list[dict]:
    time_min = _parse_bound(start) if start else datetime.now(TZ)
    time_max = _parse_bound(end, is_end=True) if end else time_min + timedelta(days=7)
    return await asyncio.to_thread(_list_sync, time_min, time_max, max_results)


async def list_events_text(start: str | None = None, end: str | None = None) -> str:
    events = await list_events(start, end)
    if not events:
        return "Keine Termine in diesem Zeitraum."
    return "\n".join(format_event(e, with_id=True) for e in events)


def _insert_sync(body: dict) -> dict:
    return _service_call(
        lambda s: s.events().insert(calendarId=CALENDAR_ID, body=body).execute()
    )


async def create_event(
    title: str,
    start: str,
    end: str | None = None,
    description: str | None = None,
    location: str | None = None,
    all_day: bool = False,
) -> str:
    if all_day:
        start_date = datetime.fromisoformat(start).date()
        # Googles Enddatum ist bei ganztägigen Terminen exklusiv.
        end_date = (datetime.fromisoformat(end).date() if end else start_date) + timedelta(days=1)
        body = {"start": {"date": start_date.isoformat()}, "end": {"date": end_date.isoformat()}}
        when = f"{start_date.strftime('%d.%m.%Y')} (ganztägig)"
    else:
        start_dt = _parse_bound(start)
        end_dt = _parse_bound(end) if end else start_dt + timedelta(hours=1)
        body = {
            "start": {"dateTime": start_dt.isoformat(), "timeZone": TIMEZONE},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": TIMEZONE},
        }
        when = start_dt.strftime("%d.%m.%Y %H:%M")

    body["summary"] = title
    if description:
        body["description"] = description
    if location:
        body["location"] = location

    event = await asyncio.to_thread(_insert_sync, body)
    return f"Termin '{title}' am {when} eingetragen. [id: {event['id']}]"


def _delete_sync(event_id: str) -> None:
    _service_call(lambda s: s.events().delete(calendarId=CALENDAR_ID, eventId=event_id).execute())


async def delete_event(event_id: str) -> str:
    from googleapiclient.errors import HttpError

    try:
        await asyncio.to_thread(_delete_sync, event_id)
    except HttpError as err:
        if err.status_code in (404, 410):
            return "Diesen Termin gibt es nicht (mehr)."
        raise
    return "Termin gelöscht."
