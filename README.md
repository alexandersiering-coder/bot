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
