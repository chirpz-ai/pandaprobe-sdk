/**
 * Minimal leveled logger for the SDK (analog of Python's logging.getLogger("pandaprobe")).
 *
 * Default level is WARNING. `setDebug(true)` (driven by PANDAPROBE_DEBUG / debug
 * config) lowers it to DEBUG.
 */

export const LogLevel = {
  DEBUG: 10,
  INFO: 20,
  WARNING: 30,
  ERROR: 40,
} as const;
export type LogLevel = (typeof LogLevel)[keyof typeof LogLevel];

let currentLevel: LogLevel = LogLevel.WARNING;

export function setDebug(debug: boolean): void {
  currentLevel = debug ? LogLevel.DEBUG : LogLevel.WARNING;
}

function fmt(msg: string): string {
  return `PandaProbe: ${msg}`;
}

export const logger = {
  debug(msg: string, ...args: unknown[]): void {
    if (currentLevel <= LogLevel.DEBUG) {
      console.debug(fmt(msg), ...args);
    }
  },
  info(msg: string, ...args: unknown[]): void {
    if (currentLevel <= LogLevel.INFO) {
      console.info(fmt(msg), ...args);
    }
  },
  warning(msg: string, ...args: unknown[]): void {
    if (currentLevel <= LogLevel.WARNING) {
      console.warn(fmt(msg), ...args);
    }
  },
  error(msg: string, ...args: unknown[]): void {
    if (currentLevel <= LogLevel.ERROR) {
      console.error(fmt(msg), ...args);
    }
  },
};
