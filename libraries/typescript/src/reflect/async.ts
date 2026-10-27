import { abortError } from "../api/client.js";
export class DeadlineExceeded extends Error {
  constructor(message = "Operation deadline exceeded") {
    super(message);
    this.name = "DeadlineExceeded";
  }
}
/** Bound a caller's wait without cancelling a shared task. */
export function waitFor<T>(promise: Promise<T>, timeout?: number, signal?: AbortSignal): Promise<T> {
  if (timeout !== undefined && (typeof timeout !== "number" || !Number.isFinite(timeout) || timeout < 0))
    return Promise.reject(new TypeError("Wait timeout must be finite and nonnegative"));
  return new Promise((resolve, reject) => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const cleanup = () => {
      if (timer) clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
    };
    const abort = () => {
      cleanup();
      reject(signal?.reason ?? abortError());
    };
    promise.then(
      (value) => {
        cleanup();
        resolve(value);
      },
      (error) => {
        cleanup();
        reject(error);
      },
    );
    if (signal?.aborted) {
      abort();
      return;
    }
    signal?.addEventListener("abort", abort, { once: true });
    if (timeout !== undefined)
      timer = setTimeout(() => {
        cleanup();
        reject(new DeadlineExceeded());
      }, timeout * 1000);
  });
}
export interface Task<T> {
  promise: Promise<T>;
  done: boolean;
  value?: T;
  error?: unknown;
}
export function task<T>(promise: Promise<T>): Task<T> {
  const record: Task<T> = { promise, done: false };
  promise.then(
    (value) => {
      record.done = true;
      record.value = value;
    },
    (error) => {
      record.done = true;
      record.error = error;
    },
  );
  return record;
}
export class Semaphore {
  private active = 0;
  private waiting: Array<() => void> = [];
  constructor(private readonly limit: number) {}
  async run<T>(callback: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    signal?.throwIfAborted();
    while (this.active >= this.limit) {
      let wake!: () => void;
      const ready = new Promise<void>((r) => {
        wake = r;
        this.waiting.push(r);
      });
      try {
        await waitFor(ready, undefined, signal);
      } finally {
        this.waiting = this.waiting.filter((w) => w !== wake);
      }
    }
    signal?.throwIfAborted();
    this.active++;
    try {
      return await callback();
    } finally {
      this.active--;
      this.waiting.shift()?.();
    }
  }
}
