# skupstina-reader

Reads who is speaking on the National Assembly live stream ([@ParlamentSrbija](https://www.youtube.com/@ParlamentSrbija))
and feeds `GET https://skupstina.app/api/live/speakers`, which the site uses to mute chosen MPs.

```json
{ "segments": [{ "from": 5012.5, "to": 5630.0, "mp": 1029 }, { "from": 5630.0, "to": null, "mp": 30 }],
  "video": "abc123", "live": true, "updated": 1795526400000 }
```

`from`/`to` are seconds of YouTube stream time (what the embedded player's `getCurrentTime()` returns), `to: null`
means still speaking, `mp` is an id from `/api/mps`.

## How it works

```
cron (1/min) ── is @ParlamentSrbija live? ──▶ start container (Cloudflare Containers, 1 instance)
                                                │ yt-dlp → 720p HLS → PyAV frames every 1.5 s
                                                │ caption strip → white mask → Tesseract (srp+srp_latn)
                                                │ name line → match 250 MPs → 3 reads = speaker change
viewers ── GET /api/live/speakers ◀── Worker ◀──┘ POST timeline (Bearer READER_SECRET), on change + every 15 s
```

- **Worker** (`src/index.ts`) owns the route `skupstina.app/api/live/speakers*`, which runs in front of the main
  `skupstina` Worker, so the site needs no change. It stores the latest timeline in the container's Durable Object
  and serves it with a 2 s edge cache.
- **Container** (`container/`) exits on its own after 15 minutes without a live stream; the cron starts it again
  when the channel goes live (and every 10 minutes if YouTube's page can't be read from the Worker).
- **Detection** follows [Neleceni](https://github.com/Tpojka/neleceni) (MIT): bottom 20 % of the frame, white
  threshold 225, letter-sized shapes only, only the line above the parliamentary-group line, Cyrillic and Latin
  treated alike, diacritics ignored. A segment opens after 3 matching reads and is back-dated to the first one.
  A caption that disappears doesn't end a segment; another speaker's caption does.

## Deploy

Push to `main` → GitHub Actions runs the tests, `wrangler deploy` (builds and pushes the image), creates
`READER_SECRET` on first deploy, and smoke-tests the endpoint.

Repo secrets: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, optional `YT_COOKIES`.

## If YouTube blocks Cloudflare's servers

`GET /api/live/speakers/debug?key=…` shows `"error": "…Sign in to confirm you're not a bot…"`. Then export cookies
from a spare Google account signed in to YouTube (Netscape `cookies.txt` format, e.g. with yt-dlp's
`--cookies-from-browser` or a cookies.txt browser extension) and save the file's contents as the `YT_COOKIES` repo
secret; the next deploy passes it to the container. A proxy can be set with `YTDLP_PROXY`.

## Testing against real video

Actions → **probe reader** → Run workflow:

- `replay` reads a recording (default: the channel's latest finished stream) and prints every reading and the
  resulting timeline.
- `live` follows the current stream for a few minutes and prints what it read.

Locally: `cd container && pip install -r requirements.txt pytest && python -m pytest` (needs `tesseract-ocr-srp`
and `tesseract-ocr-srp-latn`).

## Tuning (Worker vars, passed to the container)

| Var | Default | |
|---|---|---|
| `REGION` | `0,80,90,20` | caption area: x, y, w, h in % of the frame |
| `WHITE_THRESHOLD` | `225` | caption text brightness |
| `INTERVAL` | `1.5` | seconds between reads |
| `CONFIRM` | `3` | reads needed to change speaker |
| `TESS_LANGS` | `srp+srp_latn` | |
| `TIME_OFFSET` | `0` | seconds added to stream time, if muting turns out early/late |
| `MAX_SPEECH` | `1800` | an open segment is closed after this long |
| `OFFLINE_EXIT` | `900` | container stops after this long without a stream |
