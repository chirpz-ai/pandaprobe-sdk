import { main } from "../reflect/_support/cli.js";
// This live replay hit the default 60-second provider deadline. Allow slower
// responses here without retrying requests or changing Reflect's validation.
await main("langgraph", true, { modelTimeoutMs: 120_000 });
