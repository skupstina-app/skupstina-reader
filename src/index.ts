/**
 * skupstina-reader Worker
 *
 *   GET  /api/live/speakers         -> {segments:[{from,to,mp}], video, live, updated}  (public, 2 s edge cache)
 *   POST /api/live/speakers         <- timeline pushed by the container (Bearer READER_SECRET)
 *   GET  /api/live/speakers/debug   -> container state + last readings (Bearer or ?key=READER_SECRET)
 *
 * This route sits in front of the main "skupstina" Worker (custom domain), so the site itself
 * does not change. A cron every minute starts the container when @ParlamentSrbija is live; the
 * container exits by itself after OFFLINE_EXIT seconds without a stream.
 */
import { Container, getContainer } from "@cloudflare/containers";

export interface Env {
  READER: DurableObjectNamespace<Reader>;
  DEBUG_KV?: KVNamespace; // latest debug snapshot (frame + caption strip + OCR mask) and readings
  READER_SECRET: string;
  PUSH_URL: string;
  YT_CHANNEL: string;
  YT_COOKIES?: string;
  YTDLP_PROXY?: string;
  // optional tuning, passed through to the container when set
  REGION?: string;
  WHITE_THRESHOLD?: string;
  INTERVAL?: string;
  CONFIRM?: string;
  MIN_CONFIDENCE?: string;
  TESS_LANGS?: string;
  TIME_OFFSET?: string;
  MAX_SPEECH?: string;
  MAX_HEIGHT?: string;
  OFFLINE_EXIT?: string;
}

type Seg = { from: number; to: number | null; mp: number };
type Timeline = {
  boot: string;
  video: string | null;
  live: boolean;
  updated: number;
  received: number;
  segments: Seg[];
  debug?: string; // JSON from the container, kept as text
};
type Note = { at: number; live?: boolean | null; error?: string };

const PASS = [
  "YT_CHANNEL", "YT_COOKIES", "YTDLP_PROXY", "REGION", "WHITE_THRESHOLD", "INTERVAL", "CONFIRM",
  "MIN_CONFIDENCE", "TESS_LANGS", "TIME_OFFSET", "MAX_SPEECH", "MAX_HEIGHT", "OFFLINE_EXIT",
] as const;
const PATH = "/api/live/speakers";
const ID = "main";
const STALE_MS = 3 * 60_000; // no push for this long: the container is gone

export class Reader extends Container<Env> {
  defaultPort = 8080;
  sleepAfter = "5m";
  enableInternet = true;

  constructor(ctx: DurableObjectState<{}>, env: Env) {
    super(ctx, env);
    const vars: Record<string, string> = { PUSH_URL: env.PUSH_URL, READER_SECRET: env.READER_SECRET ?? "" };
    for (const k of PASS) if (env[k]) vars[k] = env[k] as string;
    this.envVars = vars;
  }

  /** The container pushes instead of being polled, so activity is judged by its pushes. */
  override async onActivityExpired(): Promise<void> {
    const tl = await this.ctx.storage.get<Timeline>("tl");
    if (tl && Date.now() - tl.received < STALE_MS) return; // still working: keep it running
    await this.stop();
  }

  override async onStop(): Promise<void> {
    await this.ctx.storage.put("lastStop", { at: Date.now() });
  }

  override onError(error: unknown) {
    console.error("container error", error);
    return this.ctx.storage.put("lastError", { at: Date.now(), error: String(error) });
  }

  async ingest(body: Omit<Timeline, "debug" | "received"> & { debug?: unknown }): Promise<{ ok: true; previous?: { video: string; segments: Seg[] } }> {
    const prev = await this.ctx.storage.get<Timeline>("tl");
    const next: Timeline = {
      boot: String(body.boot ?? ""),
      video: body.video ?? null,
      live: !!body.live,
      updated: Number(body.updated) || Date.now(),
      received: Date.now(),
      segments: sanitize(body.segments),
      debug: body.debug === undefined ? undefined : JSON.stringify(body.debug).slice(0, 20_000),
    };
    // A restarted container (new boot id) on the same stream gets the old timeline back.
    let previous: { video: string; segments: Seg[] } | undefined;
    if (prev && prev.boot !== next.boot && prev.video && prev.video === next.video && prev.segments.length && !next.segments.length) {
      previous = { video: prev.video, segments: prev.segments };
      next.segments = prev.segments;
    }
    // Stream ended without a new one yet: keep the last stream's timeline for late viewers.
    if (!next.video && prev?.video) {
      next.video = prev.video;
      next.segments = prev.segments.map((s) => (s.to == null ? { ...s, to: s.from } : s));
    }
    await this.ctx.storage.put("tl", next);
    this.renewActivityTimeout();
    return previous ? { ok: true, previous } : { ok: true };
  }

  async timeline(): Promise<Omit<Timeline, "debug"> | null> {
    const tl = await this.ctx.storage.get<Timeline>("tl");
    if (!tl) return null;
    const { debug: _d, ...rest } = tl;
    return rest;
  }

  async status(): Promise<string> {
    const tl = await this.ctx.storage.get<Timeline>("tl");
    return JSON.stringify({
      state: await this.getState(),
      timeline: tl ? { ...tl, debug: tl.debug ? JSON.parse(tl.debug) : null } : null,
      lastCheck: await this.ctx.storage.get<Note>("lastCheck"),
      lastStart: await this.ctx.storage.get<Note>("lastStart"),
      lastError: await this.ctx.storage.get<Note>("lastError"),
    });
  }

  /** Called by the cron: start the container when the stream is (or may be) live. */
  async tick(live: boolean | null): Promise<string> {
    await this.ctx.storage.put("lastCheck", { at: Date.now(), live });
    const st = await this.getState();
    if (st.status === "running" || st.status === "healthy") return "running";
    // Unknown (YouTube's page can't be read from the Worker): start the container and let it check
    // itself with yt-dlp; after it exits for lack of a stream, wait 5 minutes before trying again.
    const lastStop = (await this.ctx.storage.get<Note>("lastStop"))?.at ?? 0;
    if (live === true || (live === null && Date.now() - lastStop > 5 * 60_000)) {
      await this.ctx.storage.put("lastStart", { at: Date.now(), live });
      await this.startAndWaitForPorts();
      return "started";
    }
    return "idle";
  }
}

function sanitize(x: unknown): Seg[] {
  if (!Array.isArray(x)) return [];
  const out: Seg[] = [];
  for (const s of x.slice(-1000)) {
    const from = Number(s?.from), mp = Number(s?.mp);
    const to = s?.to == null ? null : Number(s.to);
    if (Number.isFinite(from) && Number.isInteger(mp) && (to === null || Number.isFinite(to))) out.push({ from, to, mp });
  }
  return out;
}

/** Is the channel live? true / false, or null when YouTube didn't give a usable answer. */
export async function checkLive(channel: string): Promise<boolean | null> {
  try {
    const r = await fetch(`https://www.youtube.com/channel/${channel}/live`, {
      headers: {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
        "accept-language": "en-US,en;q=0.9",
        cookie: "SOCS=CAI; CONSENT=YES+",
      },
      cf: { cacheTtl: 0 },
    });
    if (!r.ok) return null;
    const html = await r.text();
    if (html.includes('"isLiveNow":true')) return true;
    if (html.includes('"isLiveNow":false') || html.includes('rel="canonical" href="https://www.youtube.com/channel/')) return false;
    return null;
  } catch {
    return null;
  }
}

const json = (o: unknown, status = 200, extra: Record<string, string> = {}) =>
  new Response(JSON.stringify(o), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "access-control-allow-origin": "*", ...extra },
  });

function authorized(req: Request, url: URL, secret: string): boolean {
  if (!secret) return false;
  const given = (req.headers.get("authorization") ?? "").replace(/^Bearer\s+/i, "") || url.searchParams.get("key") || "";
  const a = new TextEncoder().encode(given), b = new TextEncoder().encode(secret);
  return a.byteLength === b.byteLength && crypto.subtle.timingSafeEqual(a, b);
}

export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);
    const reader = getContainer(env.READER, ID);

    if (url.pathname === PATH && req.method === "POST") {
      if (!authorized(req, url, env.READER_SECRET)) return json({ error: "unauthorized" }, 401);
      let body: Parameters<Reader["ingest"]>[0];
      try {
        body = await req.json();
      } catch {
        return json({ error: "bad json" }, 400);
      }
      const snap = (body as { snapshot?: { t: number; jpeg: string } }).snapshot;
      delete (body as { snapshot?: unknown }).snapshot;
      if (snap && env.DEBUG_KV) {
        const bytes = Uint8Array.from(atob(snap.jpeg), (c) => c.charCodeAt(0));
        ctx.waitUntil(Promise.all([
          env.DEBUG_KV.put("snapshot.jpg", bytes, { metadata: { t: snap.t, at: Date.now() } }),
          env.DEBUG_KV.put("debug.json", JSON.stringify({ at: Date.now(), t: snap.t, video: body.video, debug: body.debug })),
        ]));
      }
      return json(await reader.ingest(body), 200, { "cache-control": "no-store" });
    }

    if (url.pathname === PATH + "/debug") {
      if (!authorized(req, url, env.READER_SECRET)) return json({ error: "unauthorized" }, 401);
      const status = JSON.parse(await reader.status());
      return json({ ...status, liveCheck: await checkLive(env.YT_CHANNEL) }, 200, { "cache-control": "no-store" });
    }

    if (url.pathname === PATH && (req.method === "GET" || req.method === "HEAD")) {
      const cache = caches.default;
      const key = new Request(url.origin + PATH);
      const hit = await cache.match(key);
      if (hit) return hit;
      const tl = await reader.timeline();
      const fresh = !!tl && Date.now() - tl.received < STALE_MS;
      const body = {
        segments: tl?.segments ?? [],
        video: tl?.video ?? null,
        live: fresh && !!tl?.live,
        updated: tl?.updated ?? null,
      };
      const res = json(body, 200, { "cache-control": "public, max-age=2" });
      ctx.waitUntil(cache.put(key, res.clone()));
      return res;
    }

    // Anything else on this route belongs to the main site.
    return fetch(req);
  },

  async scheduled(ev: ScheduledController, env: Env, ctx: ExecutionContext): Promise<void> {
    const live = await checkLive(env.YT_CHANNEL);
        const reader = getContainer(env.READER, ID);
    const r = await reader.tick(live);
    // One line per minute in Workers Logs: enough to see what the reader is doing without the secret.
    const s = JSON.parse(await reader.status());
    const d = s.timeline?.debug ?? {};
    console.log(JSON.stringify({
      liveCheck: String(live), container: r, state: s.state?.status,
      video: s.timeline?.video ?? null, streamLive: s.timeline?.live ?? null, segments: s.timeline?.segments?.length ?? 0,
      lastPushAgoS: s.timeline ? Math.round((Date.now() - s.timeline.received) / 1000) : null,
      status: d.status, error: d.error, pushError: d.push_error, mpsError: d.mps_error, streamT: d.stream_t,
      behindS: d.behind_s, speed: d.speed,
      last: Array.isArray(d.readings) ? d.readings.slice(-2) : undefined,
    }));
  },
} satisfies ExportedHandler<Env>;
