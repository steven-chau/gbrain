/**
 * Embedding Service
 * Ported from production Ruby implementation (embedding_service.rb, 190 LOC)
 *
 * Multi-provider text embeddings: OpenAI text-embedding-3-large (default) and
 * Qwen/DashScope text-embedding-v4. Both are OpenAI-compatible APIs accessed
 * through the OpenAI SDK with provider-specific base URLs and API keys.
 *
 * Provider resolution (priority order):
 *   1. Explicit `embedding.provider` in ~/.gbrain/config.json
 *   2. Env var detection: DASHSCOPE_API_KEY → qwen, OPENAI_API_KEY → openai
 *   3. Default: openai (backward compatible)
 *
 * Retry with exponential backoff (4s base, 120s cap, 5 retries).
 */

import OpenAI from 'openai';
import { loadConfig } from './config.ts';

// ── Types ────────────────────────────────────────────────────────

export type EmbeddingProvider = 'openai' | 'qwen';

export interface EmbeddingConfig {
  provider: EmbeddingProvider;
  model: string;
  dimensions: number;
  baseUrl: string;
  apiKey: string | undefined;
  batchSize: number;
  maxChars: number;
  costPer1kTokens: number;
}

// ── Provider defaults ────────────────────────────────────────────

const PROVIDER_DEFAULTS: Record<EmbeddingProvider, Omit<EmbeddingConfig, 'apiKey'>> = {
  openai: {
    provider: 'openai',
    model: 'text-embedding-3-large',
    dimensions: 1536,
    baseUrl: 'https://api.openai.com/v1',
    batchSize: 100,
    maxChars: 8000,
    costPer1kTokens: 0.00013,
  },
  qwen: {
    provider: 'qwen',
    model: 'text-embedding-v4',
    dimensions: 1536,
    baseUrl: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
    batchSize: 10,
    maxChars: 32000, // 8192 tokens * ~4 chars/token
    costPer1kTokens: 0.00002,
  },
};

const MAX_RETRIES = 5;
const BASE_DELAY_MS = 4000;
const MAX_DELAY_MS = 120000;

// ── Provider resolution ──────────────────────────────────────────

let _config: EmbeddingConfig | null = null;

function resolveEmbeddingConfig(): EmbeddingConfig {
  const fileConfig = loadConfig();
  const embConfig = fileConfig?.embedding;

  // 1. Determine provider
  let provider: EmbeddingProvider = 'openai';
  if (embConfig?.provider === 'qwen' || embConfig?.provider === 'openai') {
    provider = embConfig.provider;
  } else if (process.env.DASHSCOPE_API_KEY) {
    provider = 'qwen';
  } else if (process.env.OPENAI_API_KEY) {
    provider = 'openai';
  }

  // 2. Resolve API key
  let apiKey: string | undefined;
  if (embConfig?.api_key) {
    apiKey = embConfig.api_key;
  } else if (provider === 'qwen') {
    apiKey = process.env.DASHSCOPE_API_KEY;
  } else {
    apiKey = process.env.OPENAI_API_KEY;
  }

  const defaults = PROVIDER_DEFAULTS[provider];
  return {
    provider,
    model: embConfig?.model ?? defaults.model,
    dimensions: defaults.dimensions,
    baseUrl: embConfig?.base_url ?? defaults.baseUrl,
    apiKey,
    batchSize: defaults.batchSize,
    maxChars: defaults.maxChars,
    costPer1kTokens: defaults.costPer1kTokens,
  };
}

export function getEmbeddingConfig(): EmbeddingConfig {
  if (!_config) _config = resolveEmbeddingConfig();
  return _config;
}

// ── Client ───────────────────────────────────────────────────────

let client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!client) {
    const cfg = getEmbeddingConfig();
    client = new OpenAI({
      baseURL: cfg.baseUrl,
      apiKey: cfg.apiKey ?? '', // prevent SDK from falling back to OPENAI_API_KEY
    });
  }
  return client;
}

// ── Public helpers ───────────────────────────────────────────────

/** True when the active embedding provider has a usable API key. */
export function isEmbeddingAvailable(): boolean {
  return !!getEmbeddingConfig().apiKey;
}

export function getEmbeddingModel(): string {
  return getEmbeddingConfig().model;
}

export function getEmbeddingDimensions(): number {
  return getEmbeddingConfig().dimensions;
}

export function getEmbeddingCostPer1kTokens(): number {
  return getEmbeddingConfig().costPer1kTokens;
}

/** Compute USD cost estimate for embedding `tokens` at the active provider's rate. */
export function estimateEmbeddingCostUsd(tokens: number): number {
  return (tokens / 1000) * getEmbeddingConfig().costPer1kTokens;
}

// ── Legacy const exports (backward-compatible) ───────────────────

export const EMBEDDING_MODEL = getEmbeddingModel();
export const EMBEDDING_DIMENSIONS = getEmbeddingDimensions();
export const EMBEDDING_COST_PER_1K_TOKENS = getEmbeddingCostPer1kTokens();

// ── Core embedding functions ─────────────────────────────────────

export async function embed(text: string): Promise<Float32Array> {
  const cfg = getEmbeddingConfig();
  const truncated = text.slice(0, cfg.maxChars);
  const result = await embedBatch([truncated]);
  return result[0];
}

export interface EmbedBatchOptions {
  /**
   * Optional callback fired after each batch completes.
   * CLI wrappers tick a reporter; Minion handlers can call
   * job.updateProgress here instead of hooking the per-page callback.
   */
  onBatchComplete?: (done: number, total: number) => void;
}

export async function embedBatch(
  texts: string[],
  options: EmbedBatchOptions = {},
): Promise<Float32Array[]> {
  const cfg = getEmbeddingConfig();
  const truncated = texts.map(t => t.slice(0, cfg.maxChars));
  const results: Float32Array[] = [];

  // Process in provider-sized batches
  for (let i = 0; i < truncated.length; i += cfg.batchSize) {
    const batch = truncated.slice(i, i + cfg.batchSize);
    const batchResults = await embedBatchWithRetry(batch);
    results.push(...batchResults);
    options.onBatchComplete?.(results.length, truncated.length);
  }

  return results;
}

async function embedBatchWithRetry(texts: string[]): Promise<Float32Array[]> {
  const cfg = getEmbeddingConfig();

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const response = await getClient().embeddings.create({
        model: cfg.model,
        input: texts,
        dimensions: cfg.dimensions,
      });

      // Sort by index to maintain order
      const sorted = response.data.sort((a, b) => a.index - b.index);
      return sorted.map(d => new Float32Array(d.embedding));
    } catch (e: unknown) {
      if (attempt === MAX_RETRIES - 1) throw e;

      // Check for rate limit with Retry-After header
      let delay = exponentialDelay(attempt);

      if (e instanceof OpenAI.APIError && e.status === 429) {
        const retryAfter = e.headers?.['retry-after'];
        if (retryAfter) {
          const parsed = parseInt(retryAfter, 10);
          if (!isNaN(parsed)) {
            delay = parsed * 1000;
          }
        }
      }

      await sleep(delay);
    }
  }

  // Should not reach here
  throw new Error('Embedding failed after all retries');
}

function exponentialDelay(attempt: number): number {
  const delay = BASE_DELAY_MS * Math.pow(2, attempt);
  return Math.min(delay, MAX_DELAY_MS);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}
