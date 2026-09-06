import { BaseProvider } from '~/lib/modules/llm/base-provider';
import type { ModelInfo } from '~/lib/modules/llm/types';
import type {
  LanguageModelV1,
  LanguageModelV1CallOptions,
  LanguageModelV1Prompt,
  LanguageModelV1StreamPart,
  FinishReason as LanguageModelV1FinishReason,
} from 'ai';
import type { IProviderSetting } from '~/types/model';
import { LLMManager } from '~/lib/modules/llm/manager';
import { createScopedLogger } from '~/utils/logger';

const logger = createScopedLogger('AMPProvider');

/**
 * Resolved AMP configuration read from the environment.
 */
interface AmpConfig {
  clientId: string;
  clientSecret: string;
  authUrl: string;
  modelEndpoint: string;

  /**
   * Optional streaming counterpart of modelEndpoint. AMP/AIS mirrors Bedrock,
   * whose streaming variant is `invoke-with-response-stream`. Only set when a
   * real streaming endpoint exists (opt-in via AMP_MODEL_STREAM_ENDPOINT); when
   * absent we use chunked continuation over the non-streaming `/invoke` instead.
   */
  modelStreamEndpoint?: string;

  /** True when a real streaming endpoint is configured. */
  streamingEnabled: boolean;

  /**
   * Max output tokens requested per `/invoke` call. Keeping this small ensures
   * each request finishes well under the Autodesk gateway timeout (avoids the
   * HTTP 504); long outputs are assembled via continuation across many calls.
   */
  chunkMaxTokens: number;
}

/**
 * In-memory cache for the OAuth2 access token. Tokens are short lived, so we
 * reuse them until shortly before expiry to avoid re-authenticating on every
 * request (mirrors the reference app authenticating once at startup).
 */
interface CachedToken {
  token: string;
  expiresAt: number;
}

/** Refresh the token this many ms before its real expiry, as a safety buffer. */
const TOKEN_EXPIRY_BUFFER_MS = 60_000;

/** Fallback max output tokens when the caller does not provide a limit. */
const DEFAULT_MAX_TOKENS = 4096;

/**
 * Default per-`/invoke` output cap for chunked continuation. Small enough that a
 * single call returns before the Autodesk gateway request timeout (the cause of
 * the HTTP 504). Tunable via AMP_CHUNK_MAX_TOKENS.
 */
const DEFAULT_CHUNK_MAX_TOKENS = 1024;

/** Hard cap on continuation rounds to guard against runaway loops. */
const MAX_CONTINUATION_ROUNDS = 16;

// Module-level cache keyed by clientId+authUrl so distinct configs don't collide.
const tokenCache = new Map<string, CachedToken>();

/**
 * Rule 7 (Secure Node.js): always use HTTPS for remote calls. AMP endpoints are
 * user-configured, so we reject any non-HTTPS URL rather than silently trusting it.
 */
function assertHttpsUrl(rawUrl: string, label: string): void {
  let parsed: URL;

  try {
    parsed = new URL(rawUrl);
  } catch {
    throw new Error(`AMP provider: ${label} is not a valid URL.`);
  }

  if (parsed.protocol !== 'https:') {
    throw new Error(`AMP provider: ${label} must use HTTPS.`);
  }
}

/**
 * Decode a base64 string to UTF-8 text across runtimes (Bedrock event-stream
 * chunks wrap the Anthropic event JSON as a base64 `bytes` field).
 */
function base64ToString(b64: string): string {
  if (typeof atob === 'function') {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);

    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }

    return new TextDecoder('utf-8').decode(bytes);
  }

  return Buffer.from(b64, 'base64').toString('utf-8');
}

/**
 * Encode "id:secret" as base64 for the HTTP Basic auth header without relying on
 * Node's Buffer (this code also runs in Cloudflare Workers / browser-like runtimes).
 */
function toBasicAuth(clientId: string, clientSecret: string): string {
  const raw = `${clientId}:${clientSecret}`;

  if (typeof btoa === 'function') {
    return btoa(raw);
  }

  // Fallback for Node runtimes where btoa may be unavailable.
  return Buffer.from(raw, 'utf-8').toString('base64');
}

/**
 * OAuth2 client-credentials grant. Equivalent to the reference app's
 * get_access_token(): HTTP Basic (client id/secret) POST with
 * grant_type=client_credentials&scope=data:read -> Bearer access_token.
 */
async function getAccessToken(config: AmpConfig): Promise<string> {
  const cacheKey = `${config.clientId}@${config.authUrl}`;
  const cached = tokenCache.get(cacheKey);

  if (cached && cached.expiresAt > Date.now()) {
    return cached.token;
  }

  const response = await fetch(config.authUrl, {
    method: 'POST',
    headers: {
      Authorization: `Basic ${toBasicAuth(config.clientId, config.clientSecret)}`,
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      grant_type: 'client_credentials',
      scope: 'data:read',
    }).toString(),
  });

  if (!response.ok) {
    // Rule 5 (Secure Node.js): do not log the token/response body which may leak secrets.
    throw new Error(`AMP provider: failed to get access token (HTTP ${response.status}).`);
  }

  const data = (await response.json()) as { access_token?: string; expires_in?: number };

  if (!data.access_token) {
    throw new Error('AMP provider: auth response did not include an access_token.');
  }

  const expiresInMs = (typeof data.expires_in === 'number' ? data.expires_in : 3600) * 1000;
  tokenCache.set(cacheKey, {
    token: data.access_token,
    expiresAt: Date.now() + Math.max(expiresInMs - TOKEN_EXPIRY_BUFFER_MS, 0),
  });

  return data.access_token;
}

interface AmpMessage {
  role: 'user' | 'assistant';
  content: Array<{ type: 'text'; text: string }>;
}

/**
 * Convert the AI SDK prompt into the Bedrock-style Anthropic shape used by AMP.
 * bolt only sends system + text messages (no tools), so we extract the system
 * text separately and map user/assistant text parts to content blocks.
 */
function convertPrompt(prompt: LanguageModelV1Prompt): { system?: string; messages: AmpMessage[] } {
  const systemParts: string[] = [];
  const messages: AmpMessage[] = [];

  for (const message of prompt) {
    if (message.role === 'system') {
      systemParts.push(message.content);
      continue;
    }

    if (message.role === 'user' || message.role === 'assistant') {
      const textBlocks = message.content
        .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
        .map((part) => ({ type: 'text' as const, text: part.text }));

      if (textBlocks.length > 0) {
        messages.push({ role: message.role, content: textBlocks });
      }
    }

    // Tool / image / reasoning parts are intentionally ignored: bolt's chat flow is text-only.
  }

  return {
    system: systemParts.length > 0 ? systemParts.join('\n\n') : undefined,
    messages,
  };
}

interface AmpBedrockResponse {
  content?: Array<{ type?: string; text?: string }>;
  usage?: { input_tokens?: number; output_tokens?: number };
  stop_reason?: string;
}

function mapStopReason(stopReason?: string): LanguageModelV1FinishReason {
  switch (stopReason) {
    case 'end_turn':
    case 'stop_sequence':
      return 'stop';
    case 'max_tokens':
      return 'length';
    default:
      return 'stop';
  }
}

/** Normalised piece of information extracted from a streamed Anthropic event. */
interface AmpStreamDelta {
  textDelta?: string;
  stopReason?: string;
  inputTokens?: number;
  outputTokens?: number;
}

/**
 * Incrementally parses an AMP streaming response. AIS mirrors Bedrock, so the
 * body arrives either as SSE (`data: {json}` lines) or as an AWS event-stream
 * whose frames embed the Anthropic event JSON as a base64 `bytes` field. Both
 * carry the same Anthropic Messages streaming events, so we normalise them to
 * AmpStreamDelta regardless of transport. Only fully-received units are parsed;
 * partial trailing data is retained until the next chunk.
 */
class AmpStreamParser {
  private _buffer = '';
  private _bedrockIndex = 0;
  private _mode: 'sse' | 'bedrock' | undefined;

  // Instance-scoped regex so concurrent streams never share lastIndex state.
  private _bytesRe = /"bytes":"([A-Za-z0-9+/=]+)"/g;

  push(chunk: string): AmpStreamDelta[] {
    this._buffer += chunk;

    if (!this._mode) {
      if (this._buffer.includes('"bytes":"')) {
        this._mode = 'bedrock';
      } else if (/(^|\n)\s*data:/.test(this._buffer)) {
        this._mode = 'sse';
      } else {
        return [];
      }
    }

    return this._mode === 'bedrock' ? this._pushBedrock() : this._pushSse();
  }

  private _pushBedrock(): AmpStreamDelta[] {
    const deltas: AmpStreamDelta[] = [];
    this._bytesRe.lastIndex = this._bedrockIndex;

    let match: RegExpExecArray | null;

    while ((match = this._bytesRe.exec(this._buffer)) !== null) {
      this._bedrockIndex = this._bytesRe.lastIndex;

      try {
        const event = JSON.parse(base64ToString(match[1]));
        const delta = this._fromAnthropicEvent(event);

        if (delta) {
          deltas.push(delta);
        }
      } catch {
        // Ignore a malformed/partial frame; the next chunk may complete it.
      }
    }

    return deltas;
  }

  private _pushSse(): AmpStreamDelta[] {
    const deltas: AmpStreamDelta[] = [];
    const lastNewline = this._buffer.lastIndexOf('\n');

    if (lastNewline === -1) {
      return deltas;
    }

    const ready = this._buffer.slice(0, lastNewline);
    this._buffer = this._buffer.slice(lastNewline + 1);

    for (const rawLine of ready.split('\n')) {
      const line = rawLine.trim();

      if (!line.startsWith('data:')) {
        continue;
      }

      const payload = line.slice('data:'.length).trim();

      if (!payload || payload === '[DONE]') {
        continue;
      }

      try {
        const delta = this._fromAnthropicEvent(JSON.parse(payload));

        if (delta) {
          deltas.push(delta);
        }
      } catch {
        // Ignore non-JSON keep-alive/comment lines.
      }
    }

    return deltas;
  }

  private _fromAnthropicEvent(event: any): AmpStreamDelta | null {
    switch (event?.type) {
      case 'message_start':
        return { inputTokens: event.message?.usage?.input_tokens };
      case 'content_block_delta':
        if (event.delta?.type === 'text_delta' && typeof event.delta.text === 'string') {
          return { textDelta: event.delta.text };
        }

        return null;
      case 'message_delta':
        return {
          stopReason: event.delta?.stop_reason,
          outputTokens: event.usage?.output_tokens,
        };
      default:
        return null;
    }
  }
}

/**
 * Minimal LanguageModelV1 adapter for AMP. AMP is a non-streaming Bedrock
 * InvokeModel-style endpoint, so doStream assembles the response via chunked
 * continuation (many short `/invoke` calls) and emits each chunk progressively.
 * A real streaming path is used only when AMP_MODEL_STREAM_ENDPOINT is set.
 */
class AmpLanguageModel implements LanguageModelV1 {
  readonly specificationVersion = 'v1' as const;
  readonly provider = 'AMP';
  readonly defaultObjectGenerationMode = undefined;

  constructor(
    readonly modelId: string,
    private readonly _config: AmpConfig,
  ) {}

  /** Build the Bedrock-style Anthropic request body. */
  private _buildBody(params: {
    system?: string;
    messages: AmpMessage[];
    maxTokens: number;
    temperature?: number;
    topP?: number;
  }): Record<string, unknown> {
    // Bedrock-style Anthropic body, matching the reference app's call_model().
    const body: Record<string, unknown> = {
      anthropic_version: 'bedrock-2023-05-31',
      max_tokens: params.maxTokens,
      messages: params.messages,
    };

    if (params.system) {
      body.system = params.system;
    }

    if (typeof params.temperature === 'number') {
      body.temperature = params.temperature;
    }

    if (typeof params.topP === 'number') {
      body.top_p = params.topP;
    }

    return body;
  }

  /** A single non-streaming `/invoke` call. Returns the raw stop_reason so the
   * continuation loop can tell a `max_tokens` cutoff from a natural end. */
  private async _invokeOnce(params: {
    system?: string;
    messages: AmpMessage[];
    maxTokens: number;
    temperature?: number;
    topP?: number;
  }): Promise<{
    text: string;
    rawStopReason?: string;
    usage: { promptTokens: number; completionTokens: number };
    body: Record<string, unknown>;
  }> {
    const token = await getAccessToken(this._config);
    const body = this._buildBody(params);

    const response = await fetch(this._config.modelEndpoint, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(`AMP provider: model call failed (HTTP ${response.status}).`);
    }

    const data = (await response.json()) as AmpBedrockResponse;

    // Pick the first text block (the model may emit a thinking block first).
    const text = data.content?.find((block) => block.type === 'text')?.text ?? '';

    return {
      text,
      rawStopReason: data.stop_reason,
      usage: {
        promptTokens: data.usage?.input_tokens ?? 0,
        completionTokens: data.usage?.output_tokens ?? 0,
      },
      body,
    };
  }

  /**
   * Assemble a full response from one or more capped `/invoke` calls. Each call
   * requests at most `chunkMaxTokens`; when the model stops due to `max_tokens`
   * we re-send its output so far as an assistant prefill and continue, until it
   * finishes naturally, the token budget is exhausted, or the round cap is hit.
   * This keeps every request short enough to beat the gateway timeout (no 504).
   * `onText`, when provided, receives each new chunk for progressive display.
   */
  private async _runChunked(
    options: LanguageModelV1CallOptions,
    onText?: (text: string) => void,
  ): Promise<{
    text: string;
    finishReason: LanguageModelV1FinishReason;
    usage: { promptTokens: number; completionTokens: number };
    body: Record<string, unknown>;
  }> {
    const { system, messages } = convertPrompt(options.prompt);
    const budget = options.maxTokens ?? DEFAULT_MAX_TOKENS;
    const chunkSize = this._config.chunkMaxTokens;

    let accumulated = '';
    let remaining = budget;
    let firstBody: Record<string, unknown> | undefined;
    let promptTokens = 0;
    let completionTokens = 0;
    let lastStopReason: string | undefined;

    for (let round = 0; round < MAX_CONTINUATION_ROUNDS; round++) {
      /*
       * On continuation rounds, prefill the assistant turn with the output so
       * far so the model resumes from there. Anthropic rejects trailing
       * whitespace in a prefill, so trim the end.
       */
      const roundMessages: AmpMessage[] =
        accumulated.length > 0
          ? [...messages, { role: 'assistant', content: [{ type: 'text', text: accumulated.trimEnd() }] }]
          : messages;

      const result = await this._invokeOnce({
        system,
        messages: roundMessages,
        maxTokens: Math.max(1, Math.min(chunkSize, remaining)),
        temperature: options.temperature,
        topP: options.topP,
      });

      if (round === 0) {
        firstBody = result.body;
      }

      if (result.text.length > 0) {
        accumulated += result.text;
        onText?.(result.text);
      }

      promptTokens = result.usage.promptTokens; // last call's input footprint
      completionTokens += result.usage.completionTokens;
      lastStopReason = result.rawStopReason;
      remaining -= result.usage.completionTokens;

      // Continue only when the per-call cap truncated the output and budget remains.
      if (result.rawStopReason !== 'max_tokens' || remaining <= 0) {
        break;
      }
    }

    return {
      text: accumulated,
      finishReason: mapStopReason(lastStopReason),
      usage: { promptTokens, completionTokens },
      body: firstBody ?? {},
    };
  }

  async doGenerate(options: LanguageModelV1CallOptions) {
    const result = await this._runChunked(options);

    return {
      text: result.text,
      finishReason: result.finishReason,
      usage: result.usage,
      rawCall: {
        rawPrompt: options.prompt,
        rawSettings: result.body,
      },
    };
  }

  /** Open a streaming model call. Throws on network failure; returns the Response otherwise. */
  private async _openStream(
    options: LanguageModelV1CallOptions,
  ): Promise<{ response: Response; body: Record<string, unknown> }> {
    const token = await getAccessToken(this._config);
    const { system, messages } = convertPrompt(options.prompt);
    const body = this._buildBody({
      system,
      messages,
      maxTokens: options.maxTokens ?? DEFAULT_MAX_TOKENS,
      temperature: options.temperature,
      topP: options.topP,
    });

    const response = await fetch(this._config.modelStreamEndpoint as string, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
        Accept: 'text/event-stream, application/vnd.amazon.eventstream',
      },
      body: JSON.stringify(body),
    });

    return { response, body };
  }

  /**
   * Pseudo-stream backed by chunked continuation over the non-streaming
   * `/invoke`. Emits each continuation chunk as it arrives (progressive UI),
   * then a final finish part. This is the default path since AMP has no
   * streaming endpoint.
   */
  private _chunkedStream(options: LanguageModelV1CallOptions) {
    const { system, messages } = convertPrompt(options.prompt);
    const rawSettings = this._buildBody({
      system,
      messages,
      maxTokens: options.maxTokens ?? DEFAULT_MAX_TOKENS,
      temperature: options.temperature,
      topP: options.topP,
    });

    // Capture `this` for use inside the ReadableStream source (non-arrow start).
    const run = (onText: (text: string) => void) => this._runChunked(options, onText);

    const stream = new ReadableStream<LanguageModelV1StreamPart>({
      async start(controller) {
        try {
          const result = await run((text) => {
            controller.enqueue({ type: 'text-delta', textDelta: text });
          });

          controller.enqueue({
            type: 'finish',
            finishReason: result.finishReason,
            usage: result.usage,
          });
          controller.close();
        } catch (error) {
          controller.error(error);
        }
      },
    });

    return {
      stream,
      rawCall: { rawPrompt: options.prompt, rawSettings },
    };
  }

  async doStream(options: LanguageModelV1CallOptions) {
    // Default: AMP has no streaming endpoint -> chunked continuation pseudo-stream.
    if (!this._config.streamingEnabled) {
      return this._chunkedStream(options);
    }

    return this._streamNative(options);
  }

  /** Real streaming path, used only when AMP_MODEL_STREAM_ENDPOINT is configured. */
  private async _streamNative(options: LanguageModelV1CallOptions) {
    let opened: { response: Response; body: Record<string, unknown> };

    try {
      opened = await this._openStream(options);
    } catch (error) {
      logger.warn(
        `AMP streaming request failed to start (${error instanceof Error ? error.message : String(error)}); ` +
          'falling back to chunked continuation.',
      );
      return this._chunkedStream(options);
    }

    const { response, body } = opened;

    /*
     * If the streaming endpoint is unavailable (e.g. not deployed / wrong path),
     * discard its body and fall back to chunked continuation.
     */
    if (!response.ok || !response.body) {
      logger.warn(`AMP streaming unavailable (HTTP ${response.status}); falling back to chunked continuation.`);
      await response.body?.cancel().catch(() => undefined);

      return this._chunkedStream(options);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    const parser = new AmpStreamParser();

    let finishReason: LanguageModelV1FinishReason = 'stop';
    let promptTokens = 0;
    let completionTokens = 0;

    const stream = new ReadableStream<LanguageModelV1StreamPart>({
      async start(controller) {
        try {
          for (;;) {
            const { done, value } = await reader.read();

            if (done) {
              break;
            }

            for (const delta of parser.push(decoder.decode(value, { stream: true }))) {
              if (delta.textDelta) {
                controller.enqueue({ type: 'text-delta', textDelta: delta.textDelta });
              }

              if (delta.stopReason) {
                finishReason = mapStopReason(delta.stopReason);
              }

              if (typeof delta.inputTokens === 'number') {
                promptTokens = delta.inputTokens;
              }

              if (typeof delta.outputTokens === 'number') {
                completionTokens = delta.outputTokens;
              }
            }
          }

          controller.enqueue({
            type: 'finish',
            finishReason,
            usage: { promptTokens, completionTokens },
          });
          controller.close();
        } catch (error) {
          controller.error(error);
        }
      },
      cancel() {
        reader.cancel().catch(() => undefined);
      },
    });

    return {
      stream,
      rawCall: { rawPrompt: options.prompt, rawSettings: body },
    };
  }
}

export default class AmpProvider extends BaseProvider {
  name = 'AMP';
  getApiKeyLink = 'https://aps.autodesk.com/';
  labelForGetApiKey = 'Get Autodesk Platform Services credentials';

  // apiTokenKey lets bolt treat the provider as configured when APS_CLIENT_ID is set.
  config = {
    apiTokenKey: 'APS_CLIENT_ID',
  };

  staticModels: ModelInfo[] = [
    {
      /*
       * The AMP model is selected by the deployment behind AMP_MODEL_ENDPOINT, so
       * this name is cosmetic and is not sent in the request body.
       */
      name: 'amp-claude-sonnet',
      label: 'AMP Claude (Autodesk)',
      provider: 'AMP',
      maxTokenAllowed: 200000,
      maxCompletionTokens: 8192,
    },
  ];

  /**
   * Read AMP configuration from server env / process env / manager env.
   * Rule 4 & 8 (Secure Node.js): secrets come from the environment, never hardcoded.
   */
  private _getConfig(serverEnv?: Record<string, string>): AmpConfig {
    const manager = LLMManager.getInstance();
    const read = (key: string): string | undefined => serverEnv?.[key] || process?.env?.[key] || manager.env?.[key];

    const clientId = read('APS_CLIENT_ID');
    const clientSecret = read('APS_CLIENT_SECRET');
    const authUrl = read('AMP_AUTH_URL');
    const modelEndpoint = read('AMP_MODEL_ENDPOINT');

    const missing = [
      ['APS_CLIENT_ID', clientId],
      ['APS_CLIENT_SECRET', clientSecret],
      ['AMP_AUTH_URL', authUrl],
      ['AMP_MODEL_ENDPOINT', modelEndpoint],
    ]
      .filter(([, value]) => !value)
      .map(([key]) => key);

    if (missing.length > 0) {
      throw new Error(`AMP provider: missing required environment variable(s): ${missing.join(', ')}.`);
    }

    /*
     * Streaming is opt-in: only enabled when a real streaming endpoint is set.
     * By default AMP has no streaming, so we use chunked continuation over the
     * non-streaming `/invoke` and skip the wasted round trip to a 404 endpoint.
     */
    const modelStreamEndpoint = read('AMP_MODEL_STREAM_ENDPOINT');
    const streamingEnabled = !!modelStreamEndpoint;

    // Per-call output cap for chunked continuation (positive integer only).
    const rawChunk = Number(read('AMP_CHUNK_MAX_TOKENS'));
    const chunkMaxTokens =
      Number.isFinite(rawChunk) && rawChunk > 0 ? Math.floor(rawChunk) : DEFAULT_CHUNK_MAX_TOKENS;

    assertHttpsUrl(authUrl as string, 'AMP_AUTH_URL');
    assertHttpsUrl(modelEndpoint as string, 'AMP_MODEL_ENDPOINT');

    if (modelStreamEndpoint) {
      assertHttpsUrl(modelStreamEndpoint, 'AMP_MODEL_STREAM_ENDPOINT');
    }

    return {
      clientId: clientId as string,
      clientSecret: clientSecret as string,
      authUrl: authUrl as string,
      modelEndpoint: modelEndpoint as string,
      modelStreamEndpoint,
      streamingEnabled,
      chunkMaxTokens,
    };
  }

  getModelInstance(options: {
    model: string;
    serverEnv: Env;
    apiKeys?: Record<string, string>;
    providerSettings?: Record<string, IProviderSetting>;
  }): LanguageModelV1 {
    const { model, serverEnv } = options;
    const config = this._getConfig(this.convertEnvToRecord(serverEnv));

    logger.info(`Creating AMP model instance for "${model}"`);

    return new AmpLanguageModel(model, config);
  }
}
