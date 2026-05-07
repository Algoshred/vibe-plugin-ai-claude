/**
 * vibe-plugin-claude
 *
 * Claude AI agent provider for VibeControls Agent.
 * Implements the AIAgentProvider interface with dual-mode support:
 * - SDK mode: Uses @anthropic-ai/sdk for direct API access
 * - CLI mode: Uses the `claude` CLI with `--print` flag
 *
 * Mode auto-detection: SDK if ANTHROPIC_API_KEY is set, CLI if `claude`
 * binary is found, error if neither is available.
 */

import { Elysia } from "elysia";

// ── Locally Redeclared Interfaces ────────────────────────────────────────
// (Avoid hard dependency on @vibecontrols/agent)

type ProviderMode = "sdk" | "cli";

interface AIModelInfo {
  id: string;
  name: string;
  provider: string;
  contextWindow: number;
  maxOutputTokens: number;
  supportsVision: boolean;
  supportsStreaming: boolean;
  inputPricePerMToken: number;
  outputPricePerMToken: number;
}

interface AIProviderCapabilities {
  streaming: boolean;
  vision: boolean;
  fileAttachments: boolean;
  toolUse: boolean;
  mcpSupport: boolean;
  voiceMode: boolean;
  cancelSupport: boolean;
  modelListing: boolean;
}

interface AIFileAttachment {
  filename: string;
  mimeType: string;
  content: Buffer | string;
  size: number;
}

interface VibePlugin {
  name: string;
  version: string;
  description?: string;
  tags?: Array<
    "backend" | "frontend" | "cli" | "provider" | "adapter" | "integration"
  >;
  cliCommand?: string;
  apiPrefix?: string;
  prerequisites?: Array<{
    name: string;
    kind: "binary" | "npm" | "pip" | "cargo" | "manual";
    requiresSudo: boolean;
    description?: string;
  }>;
  createRoutes?: () => unknown;
  providers?: { ai?: AIAgentProvider; [key: string]: unknown };
  onServerStart?: (
    app: unknown,
    hostServices?: HostServices,
  ) => void | Promise<void>;
  onServerStop?: () => void | Promise<void>;
  onCliSetup?: (
    program: unknown,
    hostServices?: HostServices,
  ) => void | Promise<void>;
}

interface HostServices {
  logger?: {
    info: (source: string, msg: string) => void;
    warn: (source: string, msg: string) => void;
    error: (source: string, msg: string) => void;
    debug: (source: string, msg: string) => void;
  };
  serviceRegistry?: {
    getService: <T>(pluginName: string, serviceName: string) => T | undefined;
  };
  getConfig: (key: string) => string | undefined | Promise<string | undefined>;
}

type AISessionStatus =
  | "active"
  | "idle"
  | "processing"
  | "error"
  | "terminated";
type AILogType =
  | "input"
  | "output"
  | "thinking"
  | "event"
  | "error"
  | "metadata";

interface AISessionConfig {
  name: string;
  agentType: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
  systemPrompt?: string;
  workingDirectory?: string;
  providerConfig?: Record<string, unknown>;
}

interface AISession {
  id: string;
  name: string;
  status: AISessionStatus;
  agentType: string;
  provider: string;
  config: AISessionConfig;
  stats: AIUsageStats;
  createdAt: string;
  updatedAt: string;
}

interface AIContext {
  id: string;
  type: string;
  content: string;
  metadata?: Record<string, unknown>;
}

interface AIResponse {
  content: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  thinkingSteps?: string[];
  durationMs: number;
  metadata?: Record<string, unknown>;
}

interface AIStreamChunk {
  type: "text" | "thinking" | "error" | "done";
  content: string;
  tokensUsed?: number;
}

interface AILog {
  id: string;
  sessionId: string;
  type: AILogType;
  content: string;
  tokenCount?: number;
  model?: string;
  durationMs?: number;
  agentMetadata?: Record<string, unknown>;
  createdAt: string;
}

interface AILogFilter {
  types?: AILogType[];
  startDate?: string;
  endDate?: string;
  search?: string;
  limit?: number;
  offset?: number;
}

interface AIUsageStats {
  inputTokens: number;
  outputTokens: number;
  requestCount: number;
  estimatedCostUsd: number;
  modelBreakdown?: Record<
    string,
    { inputTokens: number; outputTokens: number; requestCount: number }
  >;
}

interface AIAgentProvider {
  readonly name: string;
  createSession(config: AISessionConfig): Promise<AISession>;
  sendPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
  ): Promise<AIResponse>;
  streamPrompt?(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
    onChunk?: (chunk: AIStreamChunk) => void,
  ): Promise<AIResponse>;
  getSessionLogs(sessionId: string, filter?: AILogFilter): Promise<AILog[]>;
  getUsageStats(sessionId: string): Promise<AIUsageStats>;
  configureSession(
    sessionId: string,
    config: Partial<AISessionConfig>,
  ): Promise<void>;
  destroySession(sessionId: string): Promise<void>;
  listSessions(): Promise<AISession[]>;
  getSessionStatus(sessionId: string): Promise<AISessionStatus>;
  healthCheck(): Promise<{ ok: boolean; message?: string }>;
  listModels?(): Promise<AIModelInfo[]>;
  cancelRequest?(sessionId: string): Promise<void>;
  getCapabilities?(): AIProviderCapabilities;
  attachFiles?(sessionId: string, files: AIFileAttachment[]): Promise<void>;
  getMode?(): ProviderMode;
  setMode?(mode: ProviderMode): void;
  getCliLaunchSpec(): {
    binary: string;
    baseArgs?: string[];
    env?: Record<string, string>;
  } | null;
  sdkOneShot(opts: {
    prompt: string;
    model?: string;
    maxTokens?: number;
    extras?: Record<string, unknown>;
  }): Promise<{ text: string; usage?: unknown }>;
}

// Log ingester interface (from ai plugin's service registry)
interface LogIngester {
  append(input: {
    sessionId: string;
    type: AILogType;
    content: string;
    tokenCount?: number;
    model?: string;
    durationMs?: number;
    agentMetadata?: Record<string, unknown>;
  }): unknown;
}

// ── Provider Adapter Interface ──────────────────────────────────────────

interface ProviderAdapter {
  readonly mode: ProviderMode;

  sendPrompt(
    prompt: string,
    config: AISessionConfig,
    signal?: AbortSignal,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }>;

  streamPrompt(
    prompt: string,
    config: AISessionConfig,
    onChunk: (chunk: AIStreamChunk) => void,
    signal?: AbortSignal,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }>;

  healthCheck(): Promise<{ ok: boolean; message?: string }>;
}

// ── Constants ───────────────────────────────────────────────────────────

const PROVIDER_NAME = "claude";
const CLI_COMMAND = "claude";
const DISPLAY_NAME = "Claude";
const DEFAULT_MODEL = "claude-sonnet-4-20250514";
const DEFAULT_MAX_TOKENS = 8192;
const API_PREFIX = `/api/ai-${PROVIDER_NAME}`;
const SUPPORTED_MODES: ProviderMode[] = ["sdk", "cli"];
const CLI_INSTALL_COMMAND = [
  "npm",
  "install",
  "-g",
  "@anthropic-ai/claude-code",
];

const CLAUDE_MODELS: AIModelInfo[] = [
  {
    id: "claude-sonnet-4-20250514",
    name: "Claude Sonnet 4",
    provider: PROVIDER_NAME,
    contextWindow: 200_000,
    maxOutputTokens: 16_384,
    supportsVision: true,
    supportsStreaming: true,
    inputPricePerMToken: 3.0,
    outputPricePerMToken: 15.0,
  },
  {
    id: "claude-opus-4-20250514",
    name: "Claude Opus 4",
    provider: PROVIDER_NAME,
    contextWindow: 200_000,
    maxOutputTokens: 32_000,
    supportsVision: true,
    supportsStreaming: true,
    inputPricePerMToken: 15.0,
    outputPricePerMToken: 75.0,
  },
  {
    id: "claude-haiku-4-5-20251001",
    name: "Claude Haiku 4.5",
    provider: PROVIDER_NAME,
    contextWindow: 200_000,
    maxOutputTokens: 8_192,
    supportsVision: true,
    supportsStreaming: true,
    inputPricePerMToken: 0.8,
    outputPricePerMToken: 4.0,
  },
];

// ── SDK Adapter ─────────────────────────────────────────────────────────

/** Anthropic SDK type aliases to avoid importing at module level */
interface AnthropicClient {
  messages: {
    create(params: Record<string, unknown>): Promise<AnthropicResponse>;
    stream(params: Record<string, unknown>): AnthropicStream;
  };
}

interface AnthropicResponse {
  content: Array<{ type: string; text?: string }>;
  model: string;
  usage: { input_tokens: number; output_tokens: number };
}

interface AnthropicStream {
  on(event: "text", cb: (text: string) => void): AnthropicStream;
  on(event: "error", cb: (err: Error) => void): AnthropicStream;
  finalMessage(): Promise<AnthropicResponse>;
}

type ClaudeAuth =
  | { type: "apiKey"; value: string }
  | { type: "authToken"; value: string };
type ClaudeAuthResolver = () => Promise<ClaudeAuth | undefined>;

class ClaudeSdkAdapter implements ProviderAdapter {
  readonly mode: ProviderMode = "sdk";
  private client: AnthropicClient | null = null;
  private resolveAuth: ClaudeAuthResolver;

  constructor(resolveAuth: ClaudeAuthResolver) {
    this.resolveAuth = resolveAuth;
  }

  private async getClient(): Promise<AnthropicClient> {
    if (this.client) return this.client;

    const auth = await this.resolveAuth();
    if (!auth) {
      throw new Error(
        "Claude SDK auth is not configured. Set ANTHROPIC_API_KEY or " +
          "ANTHROPIC_AUTH_TOKEN, or store either key in agent config " +
          "(e.g. POST /api/config { key: 'ANTHROPIC_AUTH_TOKEN', value: '...' }).",
      );
    }

    let mod: unknown;
    try {
      mod = await import("@anthropic-ai/sdk");
    } catch {
      throw new Error(
        "Failed to load @anthropic-ai/sdk. Install it with: bun add @anthropic-ai/sdk",
      );
    }

    const m = mod as { default?: unknown };
    const Anthropic = (m.default ?? mod) as new (
      opts: { apiKey: string } | { authToken: string },
    ) => AnthropicClient;
    this.client = new Anthropic(
      auth.type === "apiKey"
        ? { apiKey: auth.value }
        : { authToken: auth.value },
    );
    return this.client;
  }

  async sendPrompt(
    prompt: string,
    config: AISessionConfig,
    signal?: AbortSignal,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }> {
    const client = await this.getClient();
    const startTime = Date.now();
    const model = config.model || DEFAULT_MODEL;

    const params: Record<string, unknown> = {
      model,
      max_tokens: config.maxTokens || DEFAULT_MAX_TOKENS,
      messages: [{ role: "user", content: prompt }],
    };

    if (config.systemPrompt) {
      params["system"] = [{ type: "text", text: config.systemPrompt }];
    }

    const response = (await client.messages.create(
      params,
    )) as AnthropicResponse;
    const durationMs = Date.now() - startTime;

    const content = response.content
      .filter((block) => block.type === "text")
      .map((block) => block.text ?? "")
      .join("");

    return {
      content,
      model: response.model,
      inputTokens: response.usage.input_tokens,
      outputTokens: response.usage.output_tokens,
      durationMs,
      metadata: { provider: PROVIDER_NAME, mode: "sdk" },
    };
  }

  async streamPrompt(
    prompt: string,
    config: AISessionConfig,
    onChunk: (chunk: AIStreamChunk) => void,
    signal?: AbortSignal,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }> {
    const client = await this.getClient();
    const startTime = Date.now();
    const model = config.model || DEFAULT_MODEL;

    const params: Record<string, unknown> = {
      model,
      max_tokens: config.maxTokens || DEFAULT_MAX_TOKENS,
      messages: [{ role: "user", content: prompt }],
    };

    if (config.systemPrompt) {
      params["system"] = [{ type: "text", text: config.systemPrompt }];
    }

    const stream = client.messages.stream(params);

    stream.on("text", (text: string) => {
      onChunk({ type: "text", content: text });
    });

    stream.on("error", (err: Error) => {
      onChunk({ type: "error", content: err.message });
    });

    const finalMessage = await stream.finalMessage();
    const durationMs = Date.now() - startTime;

    onChunk({ type: "done", content: "" });

    const content = finalMessage.content
      .filter((block) => block.type === "text")
      .map((block) => block.text ?? "")
      .join("");

    return {
      content,
      model: finalMessage.model,
      inputTokens: finalMessage.usage.input_tokens,
      outputTokens: finalMessage.usage.output_tokens,
      durationMs,
      metadata: { provider: PROVIDER_NAME, mode: "sdk" },
    };
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    try {
      await this.getClient();
      return {
        ok: true,
        message: `${DISPLAY_NAME} SDK ready (API key configured)`,
      };
    } catch (err) {
      return {
        ok: false,
        message:
          err instanceof Error ? err.message : "SDK initialization failed",
      };
    }
  }
}

// ── CLI Adapter ─────────────────────────────────────────────────────────

class ClaudeCliAdapter implements ProviderAdapter {
  readonly mode: ProviderMode = "cli";

  private buildCliArgs(config: AISessionConfig, prompt: string): string[] {
    const args: string[] = [];
    if (config.model) args.push("--model", config.model);
    if (config.maxTokens) args.push("--max-tokens", String(config.maxTokens));
    if (config.systemPrompt) args.push("--system-prompt", config.systemPrompt);
    args.push("--print", prompt);
    return args;
  }

  async sendPrompt(
    prompt: string,
    config: AISessionConfig,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }> {
    const startTime = Date.now();
    const args = this.buildCliArgs(config, prompt);

    const proc = Bun.spawn([CLI_COMMAND, ...args], {
      stdout: "pipe",
      stderr: "pipe",
      cwd: config.workingDirectory || process.cwd(),
      timeout: (config.providerConfig?.["timeoutMs"] as number) || 300_000,
    });

    const stdout = await new Response(proc.stdout).text();
    const stderr = await new Response(proc.stderr).text();
    const exitCode = await proc.exited;
    const durationMs = Date.now() - startTime;

    if (exitCode !== 0 && !stdout) {
      throw new Error(
        `${DISPLAY_NAME} CLI exited with code ${exitCode}: ${stderr}`,
      );
    }

    const content = stdout.trim() || stderr.trim();
    // CLI does not provide real token counts; approximate from character lengths
    const inputTokens = Math.ceil(prompt.length / 4);
    const outputTokens = Math.ceil(content.length / 4);
    const model = config.model || DEFAULT_MODEL;

    return {
      content,
      model,
      inputTokens,
      outputTokens,
      durationMs,
      metadata: { exitCode, provider: PROVIDER_NAME, mode: "cli" },
    };
  }

  async streamPrompt(
    prompt: string,
    config: AISessionConfig,
    onChunk: (chunk: AIStreamChunk) => void,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }> {
    // CLI does not support true streaming; run full prompt then emit chunks
    const result = await this.sendPrompt(prompt, config);
    onChunk({ type: "text", content: result.content });
    onChunk({ type: "done", content: "" });
    return result;
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    try {
      const proc = Bun.spawnSync([CLI_COMMAND, "--version"], {
        timeout: 5000,
        stdout: "pipe",
        stderr: "ignore",
      });
      if (proc.exitCode === 0) {
        return {
          ok: true,
          message: `${DISPLAY_NAME} CLI ${proc.stdout.toString().trim()}`,
        };
      }
      return {
        ok: false,
        message: `${DISPLAY_NAME} CLI not available (exit code ${proc.exitCode})`,
      };
    } catch {
      return {
        ok: false,
        message: `${DISPLAY_NAME} CLI not installed or not in PATH`,
      };
    }
  }
}

// ── Provider Implementation ─────────────────────────────────────────────

interface ManagedSession {
  id: string;
  config: AISessionConfig;
  status: AISessionStatus;
  stats: AIUsageStats;
  abortController: AbortController | null;
  files: AIFileAttachment[];
  createdAt: string;
  updatedAt: string;
}

class ClaudeProvider implements AIAgentProvider {
  readonly name = PROVIDER_NAME;
  private sessions = new Map<string, ManagedSession>();
  private logIngester: LogIngester | null = null;
  private hostServices: HostServices | null = null;
  private activeMode: ProviderMode | null = null;
  private adapter: ProviderAdapter | null = null;
  private cachedApiKey: string | undefined;
  private cachedAuthToken: string | undefined;

  setHostServices(hs: HostServices): void {
    this.hostServices = hs;
    this.logIngester =
      hs.serviceRegistry?.getService<LogIngester>("ai", "log-ingester") ?? null;

    // Warm the cache so detectMode() can see DB-stored credentials.
    void Promise.all([
      Promise.resolve(hs.getConfig("ANTHROPIC_API_KEY")),
      Promise.resolve(hs.getConfig("ANTHROPIC_AUTH_TOKEN")),
    ])
      .then(([apiKey, authToken]) => {
        const trimmedApiKey = apiKey?.trim();
        const trimmedAuthToken = authToken?.trim();
        if (trimmedApiKey) this.cachedApiKey = trimmedApiKey;
        if (trimmedAuthToken) this.cachedAuthToken = trimmedAuthToken;
      })
      .catch(() => {});
  }

  getSupportedModes(): ProviderMode[] {
    return [...SUPPORTED_MODES];
  }

  getDisplayName(): string {
    return DISPLAY_NAME;
  }

  getPrereqApiPrefix(): string {
    return API_PREFIX;
  }

  private async resolveAuth(): Promise<ClaudeAuth | undefined> {
    const envApiKey = process.env["ANTHROPIC_API_KEY"]?.trim();
    if (envApiKey) return { type: "apiKey", value: envApiKey };

    const envAuthToken = process.env["ANTHROPIC_AUTH_TOKEN"]?.trim();
    if (envAuthToken) return { type: "authToken", value: envAuthToken };

    if (this.cachedApiKey) return { type: "apiKey", value: this.cachedApiKey };
    if (this.cachedAuthToken) {
      return { type: "authToken", value: this.cachedAuthToken };
    }

    if (this.hostServices) {
      try {
        const apiKey = (
          await this.hostServices.getConfig("ANTHROPIC_API_KEY")
        )?.trim();
        if (apiKey) {
          this.cachedApiKey = apiKey;
          return { type: "apiKey", value: apiKey };
        }

        const authToken = (
          await this.hostServices.getConfig("ANTHROPIC_AUTH_TOKEN")
        )?.trim();
        if (authToken) {
          this.cachedAuthToken = authToken;
          return { type: "authToken", value: authToken };
        }
      } catch {
        return undefined;
      }
    }
    return undefined;
  }

  // ── Mode Management ──────────────────────────────────────────────────

  getMode(): ProviderMode {
    if (this.activeMode) return this.activeMode;
    return this.detectMode();
  }

  setMode(mode: ProviderMode): void {
    if (!SUPPORTED_MODES.includes(mode)) {
      throw new Error(`${DISPLAY_NAME} does not support ${mode} mode`);
    }
    this.activeMode = mode;
    this.adapter = null; // Force re-creation on next use
    this.log("info", `Mode explicitly set to: ${mode}`);
  }

  private detectMode(): ProviderMode {
    if (
      process.env["ANTHROPIC_API_KEY"]?.trim() ||
      process.env["ANTHROPIC_AUTH_TOKEN"]?.trim() ||
      this.cachedApiKey ||
      this.cachedAuthToken
    ) {
      return "sdk";
    }

    try {
      // Cross-platform: `which` on POSIX, `where.exe` on Windows.
      const finder = process.platform === "win32" ? "where.exe" : "which";
      const proc = Bun.spawnSync([finder, CLI_COMMAND], {
        timeout: 3000,
        stdout: "pipe",
        stderr: "ignore",
      });
      if (proc.exitCode === 0) return "cli";
    } catch {
      // CLI not found
    }

    // Default to SDK mode; healthCheck will report the actual failure
    return "sdk";
  }

  private getAdapter(): ProviderAdapter {
    if (this.adapter) return this.adapter;

    const mode = this.getMode();
    this.adapter =
      mode === "sdk"
        ? new ClaudeSdkAdapter(() => this.resolveAuth())
        : new ClaudeCliAdapter();
    this.activeMode = mode;
    this.log("info", `Adapter initialized in ${mode} mode`);
    return this.adapter;
  }

  // ── Session Management ───────────────────────────────────────────────

  async createSession(config: AISessionConfig): Promise<AISession> {
    const id =
      (config.providerConfig?.["sessionId"] as string) || crypto.randomUUID();
    const now = new Date().toISOString();

    // If session already exists in memory, return it
    const existing = this.sessions.get(id);
    if (existing) {
      existing.status = "active";
      existing.updatedAt = now;
      return this.toAISession(existing);
    }

    const session: ManagedSession = {
      id,
      config,
      status: "active",
      stats: {
        inputTokens: 0,
        outputTokens: 0,
        requestCount: 0,
        estimatedCostUsd: 0,
      },
      abortController: null,
      files: [],
      createdAt: now,
      updatedAt: now,
    };

    this.sessions.set(id, session);
    this.log("info", `Session created: ${id} (${config.name})`);

    return this.toAISession(session);
  }

  async sendPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
  ): Promise<AIResponse> {
    const session = this.getSession(sessionId);
    session.status = "processing";
    session.updatedAt = new Date().toISOString();

    const abortController = new AbortController();
    session.abortController = abortController;

    const fullPrompt = this.buildFullPrompt(prompt, context, session.files);

    this.logIngester?.append({
      sessionId,
      type: "input",
      content: prompt,
    });

    try {
      const adapter = this.getAdapter();
      const result = await adapter.sendPrompt(
        fullPrompt,
        session.config,
        abortController.signal,
      );

      this.updateSessionStats(session, result.inputTokens, result.outputTokens);

      this.logIngester?.append({
        sessionId,
        type: "output",
        content: result.content,
        tokenCount: result.outputTokens,
        model: result.model,
        durationMs: result.durationMs,
      });

      return {
        content: result.content,
        model: result.model,
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        durationMs: result.durationMs,
        metadata: result.metadata,
      };
    } catch (err) {
      session.status = "error";
      session.updatedAt = new Date().toISOString();

      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      this.logIngester?.append({
        sessionId,
        type: "error",
        content: errorMsg,
      });

      throw err;
    } finally {
      session.abortController = null;
    }
  }

  async streamPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
    onChunk?: (chunk: AIStreamChunk) => void,
  ): Promise<AIResponse> {
    const session = this.getSession(sessionId);
    session.status = "processing";
    session.updatedAt = new Date().toISOString();

    const abortController = new AbortController();
    session.abortController = abortController;

    const fullPrompt = this.buildFullPrompt(prompt, context, session.files);

    this.logIngester?.append({
      sessionId,
      type: "input",
      content: prompt,
    });

    try {
      const adapter = this.getAdapter();
      const chunkHandler = onChunk ?? ((_c: AIStreamChunk) => {});

      const result = await adapter.streamPrompt(
        fullPrompt,
        session.config,
        chunkHandler,
        abortController.signal,
      );

      this.updateSessionStats(session, result.inputTokens, result.outputTokens);

      this.logIngester?.append({
        sessionId,
        type: "output",
        content: result.content,
        tokenCount: result.outputTokens,
        model: result.model,
        durationMs: result.durationMs,
      });

      return {
        content: result.content,
        model: result.model,
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        durationMs: result.durationMs,
        metadata: result.metadata,
      };
    } catch (err) {
      session.status = "error";
      session.updatedAt = new Date().toISOString();

      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      this.logIngester?.append({
        sessionId,
        type: "error",
        content: errorMsg,
      });

      throw err;
    } finally {
      session.abortController = null;
    }
  }

  // ── Extended Methods ─────────────────────────────────────────────────

  async listModels(): Promise<AIModelInfo[]> {
    return [...CLAUDE_MODELS];
  }

  async cancelRequest(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);

    if (session.abortController) {
      session.abortController.abort();
      session.abortController = null;
      session.status = "active";
      session.updatedAt = new Date().toISOString();
      this.log("info", `Request cancelled for session: ${sessionId}`);
    }
  }

  getCapabilities(): AIProviderCapabilities {
    const mode = this.getMode();
    return {
      streaming: mode === "sdk",
      vision: mode === "sdk",
      fileAttachments: true,
      toolUse: false,
      mcpSupport: false,
      voiceMode: false,
      cancelSupport: mode === "sdk",
      modelListing: true,
    };
  }

  async attachFiles(
    sessionId: string,
    files: AIFileAttachment[],
  ): Promise<void> {
    const session = this.getSession(sessionId);
    session.files.push(...files);
    session.updatedAt = new Date().toISOString();
    this.log(
      "debug",
      `Attached ${files.length} file(s) to session ${sessionId}`,
    );
  }

  // ── Standard Methods ─────────────────────────────────────────────────

  async getSessionLogs(
    _sessionId: string,
    _filter?: AILogFilter,
  ): Promise<AILog[]> {
    return [];
  }

  async getUsageStats(sessionId: string): Promise<AIUsageStats> {
    const session = this.sessions.get(sessionId);
    return (
      session?.stats ?? {
        inputTokens: 0,
        outputTokens: 0,
        requestCount: 0,
        estimatedCostUsd: 0,
      }
    );
  }

  async configureSession(
    sessionId: string,
    config: Partial<AISessionConfig>,
  ): Promise<void> {
    const session = this.getSession(sessionId);
    Object.assign(session.config, config);
    session.updatedAt = new Date().toISOString();
  }

  async destroySession(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (session) {
      if (session.abortController) {
        session.abortController.abort();
        session.abortController = null;
      }
      session.status = "terminated";
      session.files = [];
      session.updatedAt = new Date().toISOString();
      this.log("info", `Session terminated: ${sessionId}`);
    }
  }

  async listSessions(): Promise<AISession[]> {
    return Array.from(this.sessions.values()).map((s) => this.toAISession(s));
  }

  async getSessionStatus(sessionId: string): Promise<AISessionStatus> {
    return this.sessions.get(sessionId)?.status ?? "terminated";
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    const adapter = this.getAdapter();
    return adapter.healthCheck();
  }

  // ── `vibe ai run` / `vibe ai sdk` integration ────────────────────────

  getCliLaunchSpec(): {
    binary: string;
    baseArgs?: string[];
    env?: Record<string, string>;
  } | null {
    const env: Record<string, string> = {};
    const apiKey =
      process.env["ANTHROPIC_API_KEY"]?.trim() || this.cachedApiKey;
    const authToken =
      process.env["ANTHROPIC_AUTH_TOKEN"]?.trim() || this.cachedAuthToken;
    if (apiKey) env["ANTHROPIC_API_KEY"] = apiKey;
    if (authToken) env["ANTHROPIC_AUTH_TOKEN"] = authToken;
    return { binary: CLI_COMMAND, env };
  }

  async sdkOneShot(opts: {
    prompt: string;
    model?: string;
    maxTokens?: number;
    extras?: Record<string, unknown>;
  }): Promise<{ text: string; usage?: unknown }> {
    const adapter = new ClaudeSdkAdapter(() => this.resolveAuth());
    const config: AISessionConfig = {
      name: "vibe-ai-sdk",
      agentType: PROVIDER_NAME,
      model: opts.model ?? DEFAULT_MODEL,
      maxTokens: opts.maxTokens ?? DEFAULT_MAX_TOKENS,
      providerConfig: opts.extras,
    };
    const result = await adapter.sendPrompt(opts.prompt, config);
    return {
      text: result.content,
      usage: {
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        model: result.model,
        durationMs: result.durationMs,
      },
    };
  }

  // ── Private Helpers ──────────────────────────────────────────────────

  private getSession(sessionId: string): ManagedSession {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    if (session.status === "terminated")
      throw new Error("Session is terminated");
    return session;
  }

  private buildFullPrompt(
    prompt: string,
    context?: AIContext[],
    files?: AIFileAttachment[],
  ): string {
    let fullPrompt = prompt;

    if (context && context.length > 0) {
      const contextStr = context
        .map((c) => `--- Context (${c.type}): ---\n${c.content}`)
        .join("\n\n");
      fullPrompt = `${prompt}\n\n${contextStr}`;
    }

    if (files && files.length > 0) {
      const fileStr = files
        .map((f) => {
          const textContent =
            typeof f.content === "string"
              ? f.content
              : f.content.toString("utf-8");
          return `--- File: ${f.filename} (${f.mimeType}, ${f.size} bytes) ---\n${textContent}`;
        })
        .join("\n\n");
      fullPrompt = `${fullPrompt}\n\n${fileStr}`;
    }

    return fullPrompt;
  }

  private updateSessionStats(
    session: ManagedSession,
    inputTokens: number,
    outputTokens: number,
  ): void {
    const model = session.config.model || DEFAULT_MODEL;
    const modelInfo = CLAUDE_MODELS.find((m) => m.id === model);

    session.stats.inputTokens += inputTokens;
    session.stats.outputTokens += outputTokens;
    session.stats.requestCount += 1;

    if (modelInfo) {
      const cost =
        (inputTokens / 1_000_000) * modelInfo.inputPricePerMToken +
        (outputTokens / 1_000_000) * modelInfo.outputPricePerMToken;
      session.stats.estimatedCostUsd += cost;
    }

    if (!session.stats.modelBreakdown) {
      session.stats.modelBreakdown = {};
    }
    const breakdown = session.stats.modelBreakdown[model] ?? {
      inputTokens: 0,
      outputTokens: 0,
      requestCount: 0,
    };
    breakdown.inputTokens += inputTokens;
    breakdown.outputTokens += outputTokens;
    breakdown.requestCount += 1;
    session.stats.modelBreakdown[model] = breakdown;

    session.status = "active";
    session.updatedAt = new Date().toISOString();
  }

  private toAISession(s: ManagedSession): AISession {
    return {
      id: s.id,
      name: s.config.name,
      status: s.status,
      agentType: s.config.agentType,
      provider: PROVIDER_NAME,
      config: s.config,
      stats: s.stats,
      createdAt: s.createdAt,
      updatedAt: s.updatedAt,
    };
  }

  private log(level: "info" | "error" | "debug", msg: string): void {
    this.hostServices?.logger?.[level]?.(`${PROVIDER_NAME}-provider`, msg);
  }
}

// ── Plugin Export ────────────────────────────────────────────────────────

function getCliVersion(): string | null {
  try {
    const proc = Bun.spawnSync([CLI_COMMAND, "--version"], {
      timeout: 5000,
      stdout: "pipe",
      stderr: "ignore",
    });
    if (proc.exitCode === 0) return proc.stdout.toString().trim();
  } catch {
    // Binary not found.
  }
  return null;
}

function createPrereqsRoutes() {
  return new Elysia({ prefix: "/prereqs" })
    .get("/status", () => {
      const version = getCliVersion();
      return {
        satisfied: Boolean(version),
        missing: version
          ? []
          : [
              {
                name: CLI_COMMAND,
                kind: "npm" as const,
                requiresSudo: false,
                description: `${DISPLAY_NAME} CLI for CLI mode`,
              },
            ],
      };
    })
    .post("/install", () => {
      if (getCliVersion()) {
        return {
          ok: true,
          installed: [CLI_COMMAND],
          pendingSudo: [],
          errors: [],
        };
      }

      const proc = Bun.spawnSync(CLI_INSTALL_COMMAND, {
        timeout: 120_000,
        stdout: "pipe",
        stderr: "pipe",
      });
      if (proc.exitCode === 0) {
        return {
          ok: true,
          installed: [CLI_COMMAND],
          pendingSudo: [],
          errors: [],
        };
      }
      return {
        ok: false,
        installed: [],
        pendingSudo: [],
        errors: [
          {
            name: CLI_COMMAND,
            message:
              proc.stderr.toString().trim() ||
              `Run manually: ${CLI_INSTALL_COMMAND.join(" ")}`,
          },
        ],
      };
    });
}

const provider = new ClaudeProvider();

export const vibePlugin: VibePlugin = {
  name: "claude",
  version: "1.0.0",
  description:
    "Claude AI agent provider for VibeControls (dual-mode: SDK + CLI)",
  tags: ["provider", "integration"],
  apiPrefix: API_PREFIX,
  prerequisites: [
    {
      name: CLI_COMMAND,
      kind: "npm",
      requiresSudo: false,
      description: `${DISPLAY_NAME} CLI for CLI mode`,
    },
  ],
  providers: { ai: provider },
  createRoutes: () => createPrereqsRoutes(),

  onServerStart(_app, hostServices) {
    if (hostServices) provider.setHostServices(hostServices);
  },

  onServerStop() {
    for (const [id] of (provider as ClaudeProvider)["sessions"]) {
      provider.destroySession(id).catch(() => {});
    }
  },
};

export default vibePlugin;
