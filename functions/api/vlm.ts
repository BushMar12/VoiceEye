// Cloudflare Pages Function — Slow Lane backend.
//
// POST /api/vlm
// Body: Ollama-compatible `/api/generate` shape so the frontend can talk to
// this endpoint with zero logic changes:
//   {
//     prompt:   string,                 // required
//     images:   [base64_jpeg_no_prefix], // required, single-image
//     options?: { num_predict?: number, temperature?: number },
//     model?:   string                  // ignored; overridden by env.VLM_MODEL
//   }
// Response: `{ response: string }` — also matches Ollama's shape so
// useVLMEngine.ts reads `data.response` unchanged.
//
// The Workers AI binding (`env.AI`) is provisioned in wrangler.toml and runs
// Llama 3.2 11B Vision Instruct by default. Override with VLM_MODEL in the
// Pages project's Environment variables.

interface Env {
  AI: Ai;
  VLM_MODEL?: string;
  VLM_MAX_TOKENS?: string;
  VLM_TEMPERATURE?: string;
  // Comma-separated allow-list of Origins that may call /api/vlm.
  // Entries may use a single leading wildcard for sub-domain matching,
  // e.g. "https://voiceeye.pages.dev,https://*.voiceeye.pages.dev".
  // Set to "*" or leave unset to disable the check (useful for dev).
  VLM_ALLOWED_ORIGINS?: string;
}

interface OllamaCompatRequest {
  prompt?: string;
  images?: string[];
  options?: {
    num_predict?: number;
    temperature?: number;
  };
}

const DEFAULT_MODEL = '@cf/meta/llama-3.2-11b-vision-instruct';
const DEFAULT_MAX_TOKENS = 512;
const DEFAULT_TEMPERATURE = 0.2;

// Same-project Pages domains are always trusted (main deploy + any
// preview deploys under *.voiceeye.pages.dev). Extra origins can be
// added via the VLM_ALLOWED_ORIGINS env var in wrangler.toml or the
// Pages dashboard.
const BUILTIN_ORIGINS = [
  'https://voiceeye.pages.dev',
  'https://*.voiceeye.pages.dev',
];

function baseCorsHeaders(origin: string | null): Record<string, string> {
  return {
    // Echo back the request's Origin when we trust it, so browsers
    // accept the response. Falls back to `null` (spec-compliant noop)
    // otherwise, which effectively blocks cross-origin scripts.
    'Access-Control-Allow-Origin': origin ?? 'null',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
    Vary: 'Origin',
  };
}

function json(
  body: unknown,
  status = 200,
  origin: string | null = null,
  extra: Record<string, string> = {},
) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      ...baseCorsHeaders(origin),
      ...extra,
    },
  });
}

function matchOrigin(origin: string, pattern: string): boolean {
  if (pattern === '*') return true;
  if (pattern === origin) return true;
  // Single leading-wildcard sub-domain match, e.g. "https://*.foo.bar".
  if (pattern.includes('://*.')) {
    const scheme = pattern.split('://', 1)[0];
    const suffix = pattern.slice(pattern.indexOf('*.') + 1); // ".foo.bar"
    return (
      origin.startsWith(`${scheme}://`) &&
      origin.endsWith(suffix) &&
      origin !== `${scheme}://${suffix.slice(1)}`
    );
  }
  return false;
}

function originAllowed(origin: string | null, env: Env): boolean {
  const raw = (env.VLM_ALLOWED_ORIGINS ?? '').trim();
  const extras = raw ? raw.split(',').map((s) => s.trim()).filter(Boolean) : [];
  const patterns = [...BUILTIN_ORIGINS, ...extras];
  if (patterns.includes('*')) return true;
  if (!origin) return false;
  return patterns.some((p) => matchOrigin(origin, p));
}

function base64ToBytes(b64: string): Uint8Array {
  // Strip any accidental `data:...;base64,` prefix.
  const clean = b64.includes(',') ? b64.slice(b64.indexOf(',') + 1) : b64;
  const binary = atob(clean);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function parseIntEnv(v: string | undefined, fallback: number): number {
  if (!v) return fallback;
  const n = Number.parseInt(v, 10);
  return Number.isFinite(n) ? n : fallback;
}

function parseFloatEnv(v: string | undefined, fallback: number): number {
  if (!v) return fallback;
  const n = Number.parseFloat(v);
  return Number.isFinite(n) ? n : fallback;
}

export const onRequestOptions: PagesFunction<Env> = ({ request, env }) => {
  const origin = request.headers.get('Origin');
  if (!originAllowed(origin, env)) {
    return new Response(null, { status: 403 });
  }
  return new Response(null, { status: 204, headers: baseCorsHeaders(origin) });
};

export const onRequestPost: PagesFunction<Env> = async ({ request, env }) => {
  const origin = request.headers.get('Origin');
  if (!originAllowed(origin, env)) {
    return json(
      { error: 'Origin not allowed. Use the VoiceEye web app.' },
      403,
      null,
    );
  }

  let body: OllamaCompatRequest;
  try {
    body = (await request.json()) as OllamaCompatRequest;
  } catch {
    return json({ error: 'Body must be valid JSON.' }, 400, origin);
  }

  const prompt = body.prompt?.trim();
  const imageB64 = body.images?.[0];
  if (!prompt) return json({ error: 'Missing `prompt`.' }, 400, origin);
  if (!imageB64) return json({ error: 'Missing `images[0]` (base64 JPEG).' }, 400, origin);

  let imageBytes: Uint8Array;
  try {
    imageBytes = base64ToBytes(imageB64);
  } catch {
    return json({ error: 'images[0] is not valid base64.' }, 400, origin);
  }

  const model = env.VLM_MODEL || DEFAULT_MODEL;
  const maxTokens = body.options?.num_predict ?? parseIntEnv(env.VLM_MAX_TOKENS, DEFAULT_MAX_TOKENS);
  const temperature = body.options?.temperature ?? parseFloatEnv(env.VLM_TEMPERATURE, DEFAULT_TEMPERATURE);

  const startedAt = Date.now();
  try {
    // Workers AI's Llama Vision binding takes `image` as an array of bytes.
    // `Array.from(Uint8Array)` is fine here — the image is ≤40 KB at the
    // frontend's 384-px / 0.5-quality JPEG encode.
    const result = (await env.AI.run(model, {
      prompt,
      image: Array.from(imageBytes),
      max_tokens: maxTokens,
      temperature,
    })) as { response?: string; description?: string };

    // Llama Vision has historically used `description`; newer models return
    // `response`. Normalise to Ollama's `response` field.
    const text = (result?.response ?? result?.description ?? '').toString().trim();
    if (!text) return json({ error: 'Empty response from Workers AI.' }, 502, origin);

    return json(
      {
        response: text,
        model,
        ms: Date.now() - startedAt,
      },
      200,
      origin,
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return json({ error: `Workers AI call failed: ${message}` }, 502, origin);
  }
};
