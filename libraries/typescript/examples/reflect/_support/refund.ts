import { Outcome, ReplayCase } from "../../../src/reflect/index.js";
import { json } from "../../../src/reflect/serialization.js";
import { timestamp } from "./reporting.js";
export const REFUND_INSTRUCTIONS = `You operate Northstar's authorized refund settlement batches.
Read the order before submitting. Your task is complete only when the requested
refund is posted; explain the actual service result without claiming a rejected
refund succeeded. A batch allows one submission: rejection closes that batch
for operations review, so you cannot repair it by retrying within this task.

Search your Reflect rule filesystem for relevant refund guidance before choosing
the amount. Applicable learned policy updates take precedence over the legacy
fallback below. Do not guess policy from general retail conventions.
Legacy fallback (used when no applicable update is available): submit the order's
invoice_total_cents as amount_cents, with the order ID as idempotency_key.
Use the supplied business tools; no real customer accounts are connected.
If a task fails, report it and finish. Rule maintenance belongs in the subsequent
reflection checkpoint, where external evidence is available.
`;
export interface RefundOrder {
  id: string;
  merchandise_paid_cents: number;
  outbound_shipping_cents: number;
}
export const DISCOVERY: RefundOrder = {
  id: "INCIDENT-101",
  merchandise_paid_cents: 6500,
  outbound_shipping_cents: 700,
};
export const FOLLOW_UP: RefundOrder = { id: "FRESH-501", merchandise_paid_cents: 9250, outbound_shipping_cents: 850 };
export const refundPrompt = (order: RefundOrder) =>
  `Post the authorized standard-return refund for order ${order.id} in this batch. Use the operational runbook and any applicable learned updates, then report the actual result.`;
export const refundCases = () => [
  new ReplayCase({
    id: "shipping-1200",
    input: { id: "HELD-201", merchandise_paid_cents: 4200, outbound_shipping_cents: 1200 },
  }),
  new ReplayCase({
    id: "shipping-300",
    input: { id: "HELD-202", merchandise_paid_cents: 11300, outbound_shipping_cents: 300 },
  }),
  new ReplayCase({
    id: "protected-free-shipping",
    input: { id: "HELD-203", merchandise_paid_cents: 2700, outbound_shipping_cents: 0 },
    protected: true,
  }),
];
export class RefundBatch {
  closed = false;
  events: any[] = [];
  ledger: any[] = [];
  constructor(readonly order: RefundOrder) {}
  lookup(order_id: string): string {
    const found = order_id === this.order.id;
    this.events.push({ operation: "lookup", order_id, found });
    return json(
      found
        ? {
            ...this.order,
            invoice_total_cents: this.order.merchandise_paid_cents + this.order.outbound_shipping_cents,
            return_type: "standard",
            authorized: true,
            batch_closed: this.closed,
          }
        : { error: "Order not found" },
      { ascii: true, spaced: true, sortKeys: false },
    );
  }
  submit(order_id: string, amount_cents: number, idempotency_key: string): string {
    const attempted_at = timestamp();
    const request = { order_id, amount_cents, idempotency_key };
    let result: any;
    if (this.closed) result = { status: "closed", posted: false, reason: "Batch already closed; no further writes" };
    else {
      this.closed = true;
      if (order_id !== this.order.id || idempotency_key !== this.order.id)
        result = { status: "rejected", posted: false, reason: "Order or idempotency key mismatch" };
      else if (!Number.isSafeInteger(amount_cents) || amount_cents !== this.order.merchandise_paid_cents)
        result = {
          status: "rejected",
          posted: false,
          reason: "REFUND_AMOUNT_POLICY_MISMATCH",
          submitted_cents: amount_cents,
          required_cents: this.order.merchandise_paid_cents,
          current_policy:
            "For authorized standard returns, refund merchandise_paid_cents only. Outbound shipping is nonrefundable. invoice_total_cents includes shipping. This supersedes the legacy invoice-total fallback.",
        };
      else {
        this.ledger.push({ ...request });
        result = { status: "posted", posted: true, ...request };
      }
    }
    result.batch_closed = this.closed;
    this.events.push({ operation: "submit", attempted_at, request, result });
    return json(result, { ascii: true, spaced: true, sortKeys: false });
  }
  verify(reply: string): Outcome {
    const expected = {
      order_id: this.order.id,
      amount_cents: this.order.merchandise_paid_cents,
      idempotency_key: this.order.id,
    };
    const submissions = this.events.filter((e) => e.operation === "submit");
    const index = this.events.findIndex((e) => e.operation === "submit");
    const checks = {
      correct_ledger: JSON.stringify(this.ledger) === JSON.stringify([expected]),
      one_submission: submissions.length === 1,
      order_read_before_write: this.events.slice(0, Math.max(0, index)).some((e) => e.found === true),
      nonempty_reply: Boolean(reply.trim()),
    };
    const success = Object.values(checks).every(Boolean);
    return new Outcome({
      scores: { task_completion: Number(success) },
      success,
      evidence: { checks, expected_ledger: [expected], actual_ledger: structuredClone(this.ledger) },
    });
  }
}
