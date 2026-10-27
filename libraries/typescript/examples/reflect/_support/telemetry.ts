/** Example telemetry: preserve native reports without treating absent cost as zero. */
export async function responseText(result: any): Promise<string> {
  if (result?.is_error)
    throw new Error(
      `Claude Agent SDK failed (${result.subtype ?? "unknown"}): ${result.result ?? "No error details returned"}`,
    );
  if (typeof result === "string") return result;
  if (result?.messages) return responseText(result.messages[result.messages.length - 1]);
  for (const field of ["finalOutput", "raw", "result", "text"]) {
    const value = await result?.[field];
    if (typeof value === "string") return value;
  }
  if (Array.isArray(result)) {
    for (const item of [...result].reverse()) {
      const text = await responseText(item);
      if (text) return text;
    }
    return "";
  }
  const content = result?.content;
  if (typeof content === "string") return content;
  const parts = content?.parts ?? content;
  return Array.isArray(parts) ? parts.map((part) => part.text ?? "").join("\n") : "";
}
export async function nativeTelemetry(framework: string, result: any, seen: Set<string>) {
  let calls: number | null = null;
  let tokens: { input: number; output: number } | null = null;
  let reportedCost: number | null = null;
  const unseen = (id: unknown) => {
    if (typeof id !== "string") return true;
    if (seen.has(id)) return false;
    seen.add(id);
    return true;
  };
  const sum = (rows: any[], input: (row: any) => unknown, output: (row: any) => unknown) => {
    if (!rows.every((row) => typeof input(row) === "number" && typeof output(row) === "number")) return null;
    return {
      input: rows.reduce((n, row) => n + Number(input(row)), 0),
      output: rows.reduce((n, row) => n + Number(output(row)), 0),
    };
  };
  if (["langchain", "langgraph", "deepagents"].includes(framework)) {
    const rows = (result?.messages ?? []).filter((m: any) => m._getType?.() === "ai" && unseen(m.id));
    calls = rows.length;
    tokens = sum(
      rows,
      (r) => r.usage_metadata?.input_tokens,
      (r) => r.usage_metadata?.output_tokens,
    );
  } else if (framework === "openai-agents" && Array.isArray(result?.rawResponses)) {
    const rows = result.rawResponses.filter((r: any) => unseen(r.responseId));
    calls = rows.length;
    tokens = sum(
      rows,
      (r) => r.usage?.inputTokens,
      (r) => r.usage?.outputTokens,
    );
  } else if (framework === "vercel-ai") {
    const rows = await result?.steps;
    if (Array.isArray(rows)) {
      calls = rows.length;
      tokens = sum(
        rows,
        (r) => r.usage?.inputTokens,
        (r) => r.usage?.outputTokens,
      );
    }
  } else if (framework === "claude-agent-sdk" && Array.isArray(result)) {
    const final = [...result].reverse().find((m) => m.type === "result");
    const assistants = result.filter((m) => m.type === "assistant");
    if (assistants.every((m) => typeof m.message?.id === "string"))
      calls = assistants.filter((m) => unseen(m.message.id)).length;
    if (final?.usage)
      tokens = sum(
        [final.usage],
        (r) => r.input_tokens,
        (r) => r.output_tokens,
      );
    if (typeof final?.total_cost_usd === "number") reportedCost = final.total_cost_usd;
  }
  return {
    model_calls: calls,
    tokens,
    native_reported_cost_usd: reportedCost,
    cost_scope:
      reportedCost === null
        ? "unavailable"
        : "Native SDK query/session report; do not assume incremental billing across resumed queries.",
  };
}
