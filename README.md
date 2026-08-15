# Telegram-Chatbot (@alesie_bot)

Telegram-Bot mit LLM-Antworten. Default: Groq (kostenlos, kein Kreditkarte).
Umschaltbar auf Gemini oder OpenAI über die `.env` — alle drei sprechen das
OpenAI-Chat-Completions-Format, deshalb reicht ein Wechsel von `LLM_BASE_URL`
und `LLM_MODEL`.

## Setup

1. Groq-Key holen: https://console.groq.com/keys
2. In `.env` bei `LLM_API_KEY=` eintragen.
3. Starten:

```bash
cd "/Users/alesie/Claude Projekte/telegram_chatbot" && ./.venv/bin/python bot.py
```

Beenden mit Strg+C. Danach in Telegram `@alesie_bot` anschreiben.

## Befehle

| Befehl   | Wirkung                       |
| -------- | ----------------------------- |
| `/start` | Begrüßung + Hilfe             |
| `/help`  | dasselbe                      |
| `/reset` | Gesprächsverlauf des Chats löschen |
| `/model` | aktuelles Modell + Endpoint    |

Alles andere (normaler Text) geht ans Modell.

## Konfiguration (`.env`)

| Variable           | Bedeutung                                                        |
| ------------------ | ---------------------------------------------------------------- |
| `TELEGRAM_BOT_TOKEN` | Token von @BotFather                                            |
| `LLM_API_KEY`      | Key des gewählten Anbieters                                       |
| `LLM_BASE_URL`     | API-Endpoint (Groq / Gemini / OpenAI — siehe `.env.example`)       |
| `LLM_MODEL`        | Modellname                                                        |
| `SYSTEM_PROMPT`    | Charakter/Rolle des Bots                                          |
| `HISTORY_TURNS`    | Wie viele Runden Kontext pro Chat gemerkt werden (Default 10)      |
| `ALLOWED_USER_IDS` | Leer = jeder darf schreiben. Sonst kommagetrennte Telegram-User-IDs |
| `BOT_NAME_TRIGGER` | Name, auf den der Bot in Gruppen zusätzlich reagiert (Default `Kollege`) |
| `WEBHOOK_URL`      | Nur für Deployment (s.u.). Lokal leer lassen → Bot läuft per Polling |
| `WEBHOOK_SECRET`   | Optional, zusätzliche Absicherung des Webhooks (s.u.)             |
| `BRING_MCP_URL`    | Optional, s. u. „Bring!-Einkaufsliste"                            |
| `BRING_MCP_TOKEN`  | Optional, s. u. „Bring!-Einkaufsliste"                            |
| `VISION_MODEL`     | Modell für Bilder (Default `qwen/qwen3.6-27b`, Groq)               |
| `PDF_MAX_CHARS`    | Max. Zeichen aus einer PDF ans Modell (Default 15000)              |
| `DATABASE_URL`     | Optional, s. u. „Erinnerungen & Notizen"                          |
| `TIMEZONE`         | Zeitzone für Erinnerungen (Default `Europe/Berlin`)                |
| `RECIPE_API_KEY`   | Optional, s. u. „Vegane Rezeptvorschläge"                          |
| `PROACTIVE_INTERVAL_HOURS` | Optional, s. u. „Proaktive Vorschläge in Gruppen" (Default 0 = aus) |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Optional, s. u. „Google-Kalender"                        |
| `GOOGLE_CALENDAR_ID` | Optional, s. u. „Google-Kalender"                               |

In Gruppen antwortet der Bot nur, wenn er per `@alesie_bot` erwähnt wird, auf
seine eigene Nachricht geantwortet wird, oder `BOT_NAME_TRIGGER` im Text
vorkommt. Im privaten 1:1-Chat antwortet er auf alles.

## Bring!-Einkaufsliste

Wenn `BRING_MCP_URL` und `BRING_MCP_TOKEN` gesetzt sind, kann der Bot die
Bring!-Einkaufsliste per natürlicher Sprache lesen und schreiben ("Kollege,
setz Milch auf die Liste", "was steht noch auf der Liste?", "hak Butter ab").
Das Modell entscheidet selbst per Function Calling, wann es die Liste liest
oder ändert — dafür braucht `LLM_MODEL` Tool-Use-Unterstützung (bei Groq z. B.
`llama-3.3-70b-versatile`, der Default hier).

Die eigentliche Bring!-Anbindung läuft nicht in diesem Bot, sondern im
separaten `geteilte einkaufsliste`-Projekt (Bring-MCP-Server auf Render).
Dessen URL + Token (aus dessen Render-Dashboard, `MCP_AUTH_TOKEN`) hier
eintragen:

```
BRING_MCP_URL=https://<dein-bring-mcp-service>.onrender.com/mcp
BRING_MCP_TOKEN=<dasselbe Token wie im Bring-MCP-Server>
```

Ohne beide Variablen bleibt das Feature einfach aus, der Bot verhält sich wie
zuvor.

## Erinnerungen & Notizen

Wenn `DATABASE_URL` gesetzt ist, kann der Bot per natürlicher Sprache
Erinnerungen setzen und Notizen speichern ("Kollege, erinnere mich morgen um
9 an den Zahnarzt", "erinnere mich jeden Montag ans Müll rausbringen", "notier
dir: WLAN-Passwort ist ..."). Das Modell entscheidet per Function Calling
selbst, wann es `create_reminder`, `list_reminders`, `delete_reminder`,
`create_note`, `list_notes` oder `delete_note` aufruft — dieselbe Technik wie
bei der Bring!-Anbindung.

Erinnerungen liegen in Postgres (persistiert über Deploys hinweg, anders als
der In-Memory-Verlauf). Kostenlos ohne Kreditkarte z. B. bei
[neon.tech](https://neon.tech): Projekt anlegen, die Connection-String-URL
(„Connection string", Format `postgresql://user:pass@host/db?sslmode=require`)
kopieren und als `DATABASE_URL` eintragen. Tabellen legt der Bot beim Start
selbst an.

Ein Hintergrund-Job im Bot prüft alle 60 Sekunden auf fällige Erinnerungen und
verschickt sie automatisch. `recurrence` steuert Wiederholung: `once`
(Standard), `daily`, `weekly`.

**Wichtig auf Render Free:** Der Service schläft nach ~15 Min. Inaktivität
ein (s. u.) — dann pausiert auch der Erinnerungs-Check, bis die nächste
Nachricht ihn aufweckt (verpasste Erinnerungen kommen dann leicht verspätet
nach, gehen aber nicht verloren). Für pünktliche Erinnerungen den Service
extern wachhalten, z. B. mit einem kostenlosen Cron-Ping auf die Render-URL
alle 5–10 Min. über [cron-job.org](https://cron-job.org) oder
[UptimeRobot](https://uptimerobot.com) — der Pfad ist egal, ein einfacher
GET auf die Basis-URL reicht (auch ein 404 zählt als „aufgeweckt").

Ohne `DATABASE_URL` bleibt das Feature einfach aus, der Bot verhält sich wie
zuvor.

### Wochentag-genaue Wiederholung

Für "3x die Woche" o. Ä. `recurrence='weekly:mon,wed,fri'` verwenden (Kürzel:
`mon,tue,wed,thu,fri,sat,sun`, kommagetrennt). Das Modell setzt das
automatisch, wenn du z. B. "erinnere mich montags, mittwochs und freitags an
..." schreibst.

### Vegane Rezeptvorschläge

Wenn zusätzlich `RECIPE_API_KEY` gesetzt ist, kann der Bot wiederkehrend
frische, gesunde vegane Rezepte vorschlagen (inkl. Zutatenliste und Link) —
z. B. "erinnere mich montags, mittwochs und freitags an ein gesundes veganes
Rezept". Das Modell setzt dafür `kind='vegan_recipe'` bei `create_reminder`;
bei jeder Fälligkeit holt der Bot dann automatisch ein neues Rezept über die
[Spoonacular-API](https://spoonacular.com/food-api), gefiltert nach
`diet=vegan` und nach Gesundheitswert sortiert.

**Wiederholungsvermeidung:** Pro Erinnerung merkt sich der Bot die letzten 8
verschickten Rezept-IDs (`recent_outputs` in der DB) und schließt sie bei der
nächsten Auswahl aus — dieselbe Erinnerung wiederholt sich also nicht.

Kostenlosen Key holen: [spoonacular.com/food-api/console](https://spoonacular.com/food-api/console#Dashboard)
(kein Kreditkarte nötig, kostenloser Tarif mit täglichem Punktelimit — für
ein paar Rezepte pro Woche reichlich). Ohne `RECIPE_API_KEY` schlägt das
Anlegen einer `vegan_recipe`-Erinnerung mit einer klaren Fehlermeldung fehl,
der Rest des Bots bleibt unberührt.

## Google-Kalender

Mit `GOOGLE_SERVICE_ACCOUNT_JSON` + `GOOGLE_CALENDAR_ID` kann der Bot Termine
lesen und schreiben ("was steht diese Woche an?", "trag Zahnarzt Montag 10 Uhr
ein") und aktiv über anstehende Termine informieren:

- **Tagesüberblick**: morgens zur eingestellten Uhrzeit die Termine des Tages
- **Vorab-Hinweis**: X Minuten vor jedem Termin eine Nachricht

Beides pro Chat einstellbar per natürlicher Sprache ("gib mir morgens um 8
einen Überblick", "sag mir 30 Minuten vor jedem Termin Bescheid"). Braucht
`DATABASE_URL`, weil die Einstellung persistiert wird. Damit derselbe Termin
nicht mehrfach angekündigt wird, merkt sich der Bot verschickte Hinweise in
`calendar_notified` (Einträge älter als 7 Tage werden automatisch aufgeräumt).

**Abgrenzung zu Erinnerungen:** Der Kalender ist für echte Termine mit
Zeitfenster; `create_reminder` (s. o.) bleibt für reine Ping-Erinnerungen ohne
Kalendereintrag.

### Einrichtung (Service Account)

Bewusst **kein** OAuth: ein Service Account braucht keinen Browser-Login und
keine ablaufenden Tokens — passt zu einem Bot, der headless auf Render läuft.

1. [Google Cloud Console](https://console.cloud.google.com) → Projekt anlegen
   (oder vorhandenes wählen).
2. **APIs & Services → Library** → „Google Calendar API" suchen → **Enable**.
3. **APIs & Services → Credentials** → **Create Credentials → Service
   account** → Name vergeben → anlegen.
4. Den erstellten Service Account öffnen → Tab **Keys** → **Add Key → Create
   new key → JSON** → die Datei wird heruntergeladen.
5. Die E-Mail-Adresse des Service Accounts kopieren (Format
   `name@projekt.iam.gserviceaccount.com`).
6. In [Google Calendar](https://calendar.google.com) → Zahnrad → **Einstellungen**
   → links den gewünschten Kalender wählen → **Für bestimmte Personen freigeben**
   → Service-Account-Adresse hinzufügen, Berechtigung **„Termine ändern"**.
7. In derselben Kalender-Ansicht unten die **Kalender-ID** kopieren (beim
   Hauptkalender ist das deine Gmail-Adresse) → als `GOOGLE_CALENDAR_ID`
   eintragen.
8. Inhalt der JSON-Datei als `GOOGLE_SERVICE_ACCOUNT_JSON` setzen. Da Render
   mit mehrzeiligen Werten manchmal zickt, geht auch base64:

   ```bash
   base64 -i ~/Downloads/dein-service-account.json | pbcopy
   ```

   Der Bot erkennt automatisch, ob der Wert JSON oder base64 ist.

Ohne beide Variablen bleibt das Feature aus, der Rest des Bots läuft normal.

## Proaktive Vorschläge in Gruppen

Wenn `PROACTIVE_INTERVAL_HOURS` auf einen Wert > 0 gesetzt ist (z. B. `3`),
liest der Bot in Gruppen auch unadressierte Nachrichten passiv mit und prüft
alle X Stunden gebündelt, ob sich daraus ein Vorschlag für die Einkaufsliste,
eine Erinnerung oder eine Notiz/ein Todo ergibt — z. B. "brauchen wir noch
Tomaten" → "Soll ich Tomaten auf die Liste setzen?". Antwortet jemand darauf
(auch per Reply), verarbeitet der Bot das ganz normal über die bestehenden
Functions.

Bewusst **keine** Prüfung pro Nachricht (zu teuer, zu aufdringlich), sondern
gebündelt in Intervallen — der Bot meldet sich nur, wenn wirklich etwas
Konkretes erkennbar ist, sonst bleibt er still. Braucht mindestens `BRING_MCP_URL`
oder `DATABASE_URL`, sonst gäbe es nichts vorzuschlagen und das Feature bleibt
inaktiv. Default `0` = aus.

**Begrüßung neuer Mitglieder** läuft unabhängig davon immer: Tritt jemand der
Gruppe bei, schickt der Bot automatisch die Kurzanleitung (`/help`-Text) als
Willkommensnachricht.

## Bilder und PDFs

- **Bilder**: einfach als Foto schicken (in Gruppen mit `Kollege` in der
  Bildunterschrift oder als Reply). Läuft über ein eigenes Vision-Modell
  (`VISION_MODEL`), unabhängig vom Text-Modell — der Groq-Default
  `llama-3.3-70b-versatile` kann selbst keine Bilder lesen. Max. 4 MB.
- **PDFs**: als Dokument schicken. Text wird lokal mit `pypdf` extrahiert und
  ganz normal ans Text-Modell gegeben (funktioniert also mit jedem Modell,
  auch ohne Vision-Support). Nur für PDFs mit echter Textebene — gescannte
  PDFs ohne Text meldet der Bot als nicht lesbar.
- Weder Bild noch PDF-Volltext landen im Gesprächsverlauf (nur eine kurze
  Notiz wie „[Bild gesendet] ..."), damit spätere Nachrichten nicht ständig
  die alten Datenmengen mitschleppen.

## Hinweise

- Der Verlauf liegt nur im Arbeitsspeicher: Neustart = Verlauf weg.
- Solange `ALLOWED_USER_IDS` leer ist, kann im privaten Chat **jeder**, der
  den Bot findet, auf deine API-Kosten chatten. Eigene ID bekommst du z. B.
  über `@userinfobot`. In Gruppen greift die Allowlist nicht — dort darf
  jedes Mitglied den Bot ansprechen.
- `.env` ist in `.gitignore` — Token und Key nicht einchecken.

## Deployment auf Render (Webhooks)

Lokal läuft der Bot per Polling (kein `WEBHOOK_URL` gesetzt). Auf Render
schaltet er automatisch auf Webhooks um, sobald `RENDER_EXTERNAL_URL`
verfügbar ist — das setzt Render selbst, du musst nichts tun.

1. Repo bei GitHub anlegen und pushen (Render deployt aus einem Git-Repo):

   ```bash
   cd "/Users/alesie/Claude Projekte/telegram_chatbot"
   git init && git add . && git commit -m "Initial commit"
   # dann auf GitHub ein leeres Repo anlegen und als Remote hinzufügen, z. B.:
   git remote add origin https://github.com/<dein-user>/<repo>.git
   git push -u origin main
   ```

2. Auf [render.com](https://render.com) → **New** → **Blueprint** → das
   GitHub-Repo auswählen. Render liest `render.yaml` automatisch und legt
   den Web Service an.
3. Render fragt nach den mit `sync: false` markierten Variablen — dort
   `TELEGRAM_BOT_TOKEN`, `LLM_API_KEY` und optional `ALLOWED_USER_IDS`
   eintragen. `WEBHOOK_SECRET` generiert Render selbst.
4. Deploy abwarten. Sobald der Service läuft, ruft `bot.py` beim Start
   automatisch Telegrams `setWebhook` mit der Render-URL auf — kein
   manueller Schritt nötig.
5. Testen: Bot in Telegram anschreiben, Antwort sollte innerhalb weniger
   Sekunden kommen.

**Hinweis Free-Plan:** Render-Web-Services im kostenlosen Plan schlafen nach
~15 Min. Inaktivität ein. Die erste Nachricht nach einer Pause braucht dann
ein paar Sekunden länger (Cold Start), geht aber nicht verloren — Telegram
versucht unzugestellte Webhook-Updates automatisch erneut.

Ohne Blueprint geht's auch manuell: **New → Web Service**, Build Command
`pip install -r requirements.txt`, Start Command `python bot.py`, und die
Env-Vars aus der Tabelle oben von Hand eintragen (`WEBHOOK_URL` NICHT selbst
setzen, Render liefert `RENDER_EXTERNAL_URL` automatisch).
