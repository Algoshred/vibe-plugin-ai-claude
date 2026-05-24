/**
 * vibe-plugin-claude Provider Tests
 *
 * Tests for the ClaudeProvider class exported via the vibePlugin.
 */
import { beforeEach, describe, expect, it, mock, spyOn } from "bun:test";

// Mock @anthropic-ai/sdk before importing the plugin
mock.module("@anthropic-ai/sdk", () => {
  class MockAnthropic {
    messages = {
      create: mock(() =>
        Promise.resolve({
          content: [{ type: "text", text: "Hello from Claude" }],
          model: "claude-sonnet-4-20250514",
          usage: { input_tokens: 10, output_tokens: 20 },
        }),
      ),
      stream: mock(() => {
        const handlers: Record<string, (arg: unknown) => void> = {};
        return {
          on(event: string, cb: (arg: unknown) => void) {
            handlers[event] = cb;
            // Simulate text event immediately for 'text'
            if (event === "text") {
              setTimeout(() => cb("streamed text"), 0);
            }
            return this;
          },
          finalMessage: mock(() =>
            Promise.resolve({
              content: [{ type: "text", text: "streamed text" }],
              model: "claude-sonnet-4-20250514",
              usage: { input_tokens: 5, output_tokens: 15 },
            }),
          ),
        };
      }),
    };
  }
  return { default: MockAnthropic };
});

const { createPlugin, permissionFlags } = await import("../index.js");
const vibePlugin = createPlugin({ name: "test", dataDir: "/tmp" });

// Extract the provider from the plugin
const provider = vibePlugin.providers!.ai!;

describe("ClaudeProvider", () => {
  const sessionConfig = {
    name: "test-session",
    agentType: "claude",
    model: "claude-sonnet-4-20250514",
    maxTokens: 4096,
  };

  beforeEach(() => {
    // Ensure SDK mode is used
    process.env["ANTHROPIC_API_KEY"] = "test-key-123";
    provider.setMode!("sdk");
  });

  // ── Session Lifecycle ───────────────────────────────────────────

  describe("createSession", () => {
    it("creates a new session with generated ID", async () => {
      const session = await provider.createSession(sessionConfig);

      expect(session.id).toBeDefined();
      expect(session.name).toBe("test-session");
      expect(session.agentType).toBe("claude");
      expect(session.provider).toBe("claude");
      expect(session.status).toBe("active");
      expect(session.stats.inputTokens).toBe(0);
      expect(session.stats.outputTokens).toBe(0);
      expect(session.stats.requestCount).toBe(0);
      expect(session.createdAt).toBeDefined();
    });

    it("uses provided sessionId from providerConfig", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: "custom-id-001" },
      });

      expect(session.id).toBe("custom-id-001");
    });

    it("returns existing session if ID already exists", async () => {
      const session1 = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: "reuse-id" },
      });
      const session2 = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: "reuse-id" },
      });

      expect(session1.id).toBe(session2.id);
      expect(session2.status).toBe("active");
    });
  });

  describe("configureSession", () => {
    it("updates session config", async () => {
      const session = await provider.createSession(sessionConfig);
      await provider.configureSession(session.id, {
        model: "claude-opus-4-20250514",
      });

      // Verify by listing sessions
      const sessions = await provider.listSessions();
      const updated = sessions.find((s) => s.id === session.id);
      expect(updated?.config.model).toBe("claude-opus-4-20250514");
    });

    it("throws for non-existent session", async () => {
      await expect(
        provider.configureSession("does-not-exist", { model: "x" }),
      ).rejects.toThrow("not found");
    });
  });

  describe("destroySession", () => {
    it("terminates session and cleans up", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: "destroy-me" },
      });

      await provider.destroySession(session.id);

      const status = await provider.getSessionStatus(session.id);
      expect(status).toBe("terminated");
    });

    it("no-ops for unknown session ID", async () => {
      // Should not throw
      await provider.destroySession("nonexistent-session");
    });
  });

  describe("listSessions", () => {
    it("returns all sessions", async () => {
      const id = `list-test-${Date.now()}`;
      await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: id },
      });

      const sessions = await provider.listSessions();
      expect(sessions.length).toBeGreaterThanOrEqual(1);
      expect(sessions.some((s) => s.id === id)).toBe(true);
    });
  });

  describe("getSessionStatus", () => {
    it("returns status for existing session", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `status-${Date.now()}` },
      });

      const status = await provider.getSessionStatus(session.id);
      expect(status).toBe("active");
    });

    it("returns terminated for unknown session", async () => {
      const status = await provider.getSessionStatus("totally-unknown");
      expect(status).toBe("terminated");
    });
  });

  // ── sendPrompt ──────────────────────────────────────────────────

  describe("sendPrompt", () => {
    it("sends prompt via SDK adapter and returns response", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `prompt-${Date.now()}` },
      });

      const response = await provider.sendPrompt(session.id, "What is 2+2?");

      expect(response.content).toBe("Hello from Claude");
      expect(response.model).toBe("claude-sonnet-4-20250514");
      expect(response.inputTokens).toBe(10);
      expect(response.outputTokens).toBe(20);
      expect(response.durationMs).toBeGreaterThanOrEqual(0);
    });

    it("accumulates usage stats across multiple prompts", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `multi-prompt-${Date.now()}` },
      });

      await provider.sendPrompt(session.id, "First prompt");
      await provider.sendPrompt(session.id, "Second prompt");

      const stats = await provider.getUsageStats(session.id);
      expect(stats.inputTokens).toBe(20); // 10 + 10
      expect(stats.outputTokens).toBe(40); // 20 + 20
      expect(stats.requestCount).toBe(2);
    });

    it("throws for non-existent session", async () => {
      await expect(provider.sendPrompt("ghost", "Hello")).rejects.toThrow(
        "not found",
      );
    });

    it("throws for terminated session", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `terminated-prompt-${Date.now()}` },
      });
      await provider.destroySession(session.id);

      await expect(provider.sendPrompt(session.id, "Hello")).rejects.toThrow(
        "terminated",
      );
    });
  });

  // ── getUsageStats ───────────────────────────────────────────────

  describe("getUsageStats", () => {
    it("returns zero stats for fresh session", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `fresh-stats-${Date.now()}` },
      });

      const stats = await provider.getUsageStats(session.id);
      expect(stats.inputTokens).toBe(0);
      expect(stats.outputTokens).toBe(0);
      expect(stats.requestCount).toBe(0);
      expect(stats.estimatedCostUsd).toBe(0);
    });

    it("returns default stats for unknown session", async () => {
      const stats = await provider.getUsageStats("no-such-session");
      expect(stats.inputTokens).toBe(0);
      expect(stats.requestCount).toBe(0);
    });
  });

  // ── healthCheck ─────────────────────────────────────────────────

  describe("healthCheck", () => {
    it("returns ok when SDK is available", async () => {
      const result = await provider.healthCheck();
      expect(result.ok).toBe(true);
      expect(result.message).toContain("SDK");
    });
  });

  // ── getCapabilities ─────────────────────────────────────────────

  describe("getCapabilities", () => {
    it("returns correct capabilities for SDK mode", () => {
      provider.setMode!("sdk");
      const caps = provider.getCapabilities!();

      expect(caps.streaming).toBe(true);
      expect(caps.vision).toBe(true);
      expect(caps.fileAttachments).toBe(true);
      expect(caps.toolUse).toBe(false);
      expect(caps.mcpSupport).toBe(false);
      expect(caps.cancelSupport).toBe(true);
      expect(caps.modelListing).toBe(true);
    });

    it("returns correct capabilities for CLI mode", () => {
      provider.setMode!("cli");
      const caps = provider.getCapabilities!();

      expect(caps.streaming).toBe(false);
      expect(caps.vision).toBe(false);
      expect(caps.cancelSupport).toBe(false);

      // Restore SDK mode
      provider.setMode!("sdk");
    });
  });

  // ── getMode / setMode ───────────────────────────────────────────

  describe("getMode / setMode", () => {
    it("defaults to sdk when ANTHROPIC_API_KEY is set", () => {
      process.env["ANTHROPIC_API_KEY"] = "key";
      // Force re-detect by setting mode then clearing
      const mode = provider.getMode!();
      expect(mode).toBe("sdk");
    });

    it("allows explicit mode switching", () => {
      provider.setMode!("cli");
      expect(provider.getMode!()).toBe("cli");

      provider.setMode!("sdk");
      expect(provider.getMode!()).toBe("sdk");
    });
  });

  // ── listModels ──────────────────────────────────────────────────

  describe("listModels", () => {
    it("returns available Claude models", async () => {
      const models = await provider.listModels!();

      expect(models.length).toBeGreaterThanOrEqual(3);
      expect(models.every((m) => m.provider === "claude")).toBe(true);

      const sonnet = models.find((m) => m.id.includes("sonnet"));
      expect(sonnet).toBeDefined();
      expect(sonnet!.supportsVision).toBe(true);
      expect(sonnet!.supportsStreaming).toBe(true);
      expect(sonnet!.contextWindow).toBe(200_000);
    });

    it("returns a copy (not the internal array)", async () => {
      const models1 = await provider.listModels!();
      const models2 = await provider.listModels!();
      expect(models1).not.toBe(models2);
      expect(models1).toEqual(models2);
    });
  });

  // ── CLI mode prompt delivery (regression) ───────────────────────

  describe("CLI mode prompt delivery", () => {
    it("delivers the prompt via stdin, never as an argv positional", async () => {
      provider.setMode!("cli");
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `cli-${Date.now()}` },
      });

      let captured: { cmd: string[]; stdin: unknown } | null = null;
      const spawnSpy = spyOn(Bun, "spawn").mockImplementation(((
        cmd: string[],
        opts: { stdin?: unknown },
      ) => {
        captured = { cmd, stdin: opts?.stdin };
        return {
          stdout: new TextEncoder().encode("PONG"),
          stderr: new TextEncoder().encode(""),
          exited: Promise.resolve(0),
        };
      }) as unknown as typeof Bun.spawn);

      try {
        // A prompt beginning with "--" used to be misparsed by the claude CLI
        // as an unknown option because it was passed as an argv positional.
        const trickyPrompt =
          "--- Conversation History ---\nhello\n--- End History ---";
        const res = await provider.sendPrompt(session.id, trickyPrompt);

        expect(res.content).toBe("PONG");
        expect(captured).not.toBeNull();
        // print mode is requested via flag...
        expect(captured!.cmd).toContain("--print");
        // ...but the prompt text never appears in the argv (would break parsing)
        expect(
          captured!.cmd.some((a) => a.includes("Conversation History")),
        ).toBe(false);
        // the prompt is delivered on stdin instead
        const stdinText = new TextDecoder().decode(
          captured!.stdin as Uint8Array,
        );
        expect(stdinText.includes("Conversation History")).toBe(true);
      } finally {
        spawnSpy.mockRestore();
        provider.setMode!("sdk");
      }
    });

    it.each([
      ["plan", "plan"],
      ["acceptEdits", "acceptEdits"],
      ["fullAuto", "bypassPermissions"],
    ] as const)(
      "spawns claude with --permission-mode %s → %s",
      async (permissionMode, expectedFlag) => {
        provider.setMode!("cli");
        const session = await provider.createSession({
          ...sessionConfig,
          permissionMode,
          providerConfig: { sessionId: `cli-pm-${permissionMode}-${Date.now()}` },
        });

        let captured: string[] | null = null;
        const spawnSpy = spyOn(Bun, "spawn").mockImplementation(((
          cmd: string[],
        ) => {
          captured = cmd;
          return {
            stdout: new TextEncoder().encode("OK"),
            stderr: new TextEncoder().encode(""),
            exited: Promise.resolve(0),
          };
        }) as unknown as typeof Bun.spawn);

        try {
          await provider.sendPrompt(session.id, "hello");
          expect(captured).not.toBeNull();
          // the exact argv handed to Bun.spawn is what becomes the process
          // cmdline — assert the permission flag pair is present and adjacent
          const idx = captured!.indexOf("--permission-mode");
          expect(idx).toBeGreaterThanOrEqual(0);
          expect(captured![idx + 1]).toBe(expectedFlag);
        } finally {
          spawnSpy.mockRestore();
          provider.setMode!("sdk");
        }
      },
    );
  });

  // ── cancelRequest ───────────────────────────────────────────────

  describe("cancelRequest", () => {
    it("throws for unknown session", async () => {
      await expect(provider.cancelRequest!("missing")).rejects.toThrow(
        "not found",
      );
    });
  });

  // ── attachFiles ─────────────────────────────────────────────────

  describe("attachFiles", () => {
    it("attaches files to an existing session", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `files-${Date.now()}` },
      });

      await provider.attachFiles!(session.id, [
        {
          filename: "test.txt",
          mimeType: "text/plain",
          content: "hello",
          size: 5,
        },
      ]);

      // No error means success
    });

    it("throws for non-existent session", async () => {
      await expect(
        provider.attachFiles!("none", [
          { filename: "f.txt", mimeType: "text/plain", content: "x", size: 1 },
        ]),
      ).rejects.toThrow("not found");
    });
  });
});

describe("permissionFlags", () => {
  it("maps plan", () => {
    expect(permissionFlags("plan")).toEqual(["--permission-mode", "plan"]);
  });

  it("maps acceptEdits", () => {
    expect(permissionFlags("acceptEdits")).toEqual([
      "--permission-mode",
      "acceptEdits",
    ]);
  });

  it("maps fullAuto to bypassPermissions", () => {
    expect(permissionFlags("fullAuto")).toEqual([
      "--permission-mode",
      "bypassPermissions",
    ]);
  });

  it("falls back to acceptEdits for undefined / unknown", () => {
    expect(permissionFlags(undefined)).toEqual([
      "--permission-mode",
      "acceptEdits",
    ]);
    // unknown string should not throw and should not be the most-permissive
    expect(permissionFlags("bogus" as never)).toEqual([
      "--permission-mode",
      "acceptEdits",
    ]);
  });
});
