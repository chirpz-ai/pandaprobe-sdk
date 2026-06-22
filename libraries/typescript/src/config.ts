/**
 * Configuration management for the PandaProbe SDK.
 *
 * Reads from environment variables with the `PANDAPROBE_` prefix, with explicit
 * options taking precedence.
 */

import { setDebug } from "./logger.js";

function env(name: string): string | undefined {
  return process.env[name];
}

function envBool(name: string, fallback: boolean): boolean {
  const val = process.env[name];
  if (val === undefined) {
    return fallback;
  }
  return ["true", "1", "yes"].includes(val.toLowerCase());
}

function envInt(name: string, fallback: number): number {
  const val = process.env[name];
  if (val === undefined) {
    return fallback;
  }
  const parsed = Number.parseInt(val, 10);
  return Number.isNaN(parsed) ? fallback : parsed;
}

function envFloat(name: string, fallback: number): number {
  const val = process.env[name];
  if (val === undefined) {
    return fallback;
  }
  const parsed = Number.parseFloat(val);
  return Number.isNaN(parsed) ? fallback : parsed;
}

const DEFAULT_ENDPOINT = "https://api.pandaprobe.com";

export interface SdkConfig {
  readonly apiKey: string | undefined;
  readonly projectName: string | undefined;
  readonly endpoint: string;
  readonly environment: string | undefined;
  readonly release: string | undefined;
  readonly enabled: boolean;
  readonly batchSize: number;
  readonly flushInterval: number;
  readonly maxQueueSize: number;
  readonly debug: boolean;
}

export interface ConfigOptions {
  apiKey?: string;
  projectName?: string;
  endpoint?: string;
  environment?: string;
  release?: string;
  enabled?: boolean;
  batchSize?: number;
  flushInterval?: number;
  maxQueueSize?: number;
  debug?: boolean;
}

/** Build an `SdkConfig` by merging explicit options over env vars. */
export function resolveConfig(options: ConfigOptions = {}): SdkConfig {
  const config: SdkConfig = Object.freeze({
    apiKey: options.apiKey ?? env("PANDAPROBE_API_KEY"),
    projectName: options.projectName ?? env("PANDAPROBE_PROJECT_NAME"),
    endpoint: options.endpoint ?? env("PANDAPROBE_ENDPOINT") ?? DEFAULT_ENDPOINT,
    environment: options.environment ?? env("PANDAPROBE_ENVIRONMENT"),
    release: options.release ?? env("PANDAPROBE_RELEASE"),
    enabled: options.enabled ?? envBool("PANDAPROBE_ENABLED", true),
    batchSize: options.batchSize ?? envInt("PANDAPROBE_BATCH_SIZE", 10),
    flushInterval: options.flushInterval ?? envFloat("PANDAPROBE_FLUSH_INTERVAL", 5.0),
    maxQueueSize: options.maxQueueSize ?? envInt("PANDAPROBE_MAX_QUEUE_SIZE", 1000),
    debug: options.debug ?? envBool("PANDAPROBE_DEBUG", false),
  });

  if (config.enabled && !config.apiKey) {
    throw new Error("PandaProbe API key is required. Set PANDAPROBE_API_KEY or pass apiKey to Client / init().");
  }
  if (config.enabled && !config.projectName) {
    throw new Error(
      "PandaProbe project name is required. Set PANDAPROBE_PROJECT_NAME or pass projectName to Client / init().",
    );
  }
  if (config.debug) {
    setDebug(true);
  }

  return config;
}

/** Read PANDAPROBE_ENABLED for auto-init gating (mirrors Python `_env_bool`). */
export function envEnabled(): boolean {
  return envBool("PANDAPROBE_ENABLED", true);
}
