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

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Max-Age': '86400',
};

function json(body: unknown, status = 200, extra: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      ...CORS_HEADERS,
      ...extra,
    },
  });
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

export const onRequestOptions: PagesFunction<Env> = () =>
  new Response(null, { status: 204, headers: CORS_HEADERS });

export const onRequestPost: PagesFunction<Env> = async ({ request, env }) => {
  let body: OllamaCompatRequest;
  try {
    body = (await request.json()) as OllamaCompatRequest;
  } catch {
    return json({ error: 'Body must be valid JSON.' }, 400);
  }

  const prompt = body.prompt?.trim();
  const imageB64 = body.images?.[0];
  if (!prompt) return json({ error: 'Missing `prompt`.' }, 400);
  if (!imageB64) return json({ error: 'Missing `images[0]` (base64 JPEG).' }, 400);

  let imageBytes: Uint8Array;
  try {
    imageBytes = base64ToBytes(imageB64);
  } catch {
    return json({ error: 'images[0] is not valid base64.' }, 400);
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
    if (!text) return json({ error: 'Empty response from Workers AI.' }, 502);

    return json({
      response: text,
      model,
      ms: Date.now() - startedAt,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return json({ error: `Workers AI call failed: ${message}` }, 502);
  }
};
