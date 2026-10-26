"""Controlled stale-runbook incident, with a separate business-state oracle.

This is a fictional refund batch, not a benchmark of naturally occurring errors.
The prompt contains an obsolete fallback. The service's policy is constant across
discovery, both replay arms, and the fresh follow-up; it never inspects guidance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from pandaprobe.reflect import Outcome, ReplayCase

INSTRUCTIONS = """You operate Northstar's authorized refund settlement batches.
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
"""


@dataclass(frozen=True)
class RefundOrder:
    id: str
    merchandise_paid_cents: int
    outbound_shipping_cents: int

    @property
    def prompt(self) -> str:
        return (
            f"Post the authorized standard-return refund for order {self.id} in this batch. "
            "Use the operational runbook and any applicable learned updates, then report the actual result."
        )


DISCOVERY = RefundOrder("INCIDENT-101", 6500, 700)
FOLLOW_UP = RefundOrder("FRESH-501", 9250, 850)


def replay_cases() -> list[ReplayCase]:
    return [
        ReplayCase("shipping-1200", asdict(RefundOrder("HELD-201", 4200, 1200))),
        ReplayCase("shipping-300", asdict(RefundOrder("HELD-202", 11300, 300))),
        # Both policies agree here. Overgeneralizing the correction must not break this case.
        ReplayCase("protected-free-shipping", asdict(RefundOrder("HELD-203", 2700, 0)), protected=True),
    ]


class RefundBatch:
    """A single immutable-policy batch with an append-only simulated ledger.

    Deliberate fault fixture: the deployed service excludes outbound shipping,
    but the agent's fallback runbook predates that policy. A rejected batch is
    quarantined, which makes recovery within the same task impossible. Every arm
    has this identical constraint, even after guidance is accepted.
    """

    def __init__(self, order: RefundOrder):
        self.order = order
        self.closed = False
        self.events: list[dict] = []
        self.ledger: list[dict] = []

    @property
    def functions(self):
        return [self.lookup_refund_order, self.submit_refund]

    def lookup_refund_order(self, order_id: str) -> str:
        """Read the authorized standard-return order and its payment breakdown."""
        found = order_id == self.order.id
        self.events.append({"operation": "lookup", "order_id": order_id, "found": found})
        if not found:
            return json.dumps({"error": "Order not found"})
        return json.dumps(
            {
                **asdict(self.order),
                "invoice_total_cents": self.order.merchandise_paid_cents + self.order.outbound_shipping_cents,
                "return_type": "standard",
                "authorized": True,
                "batch_closed": self.closed,
            }
        )

    def submit_refund(self, order_id: str, amount_cents: int, idempotency_key: str) -> str:
        """Submit this batch's one authorized refund; a rejected submission closes the batch."""
        attempted_at = datetime.now(timezone.utc).isoformat()
        request = {"order_id": order_id, "amount_cents": amount_cents, "idempotency_key": idempotency_key}
        if self.closed:
            result = {"status": "closed", "posted": False, "reason": "Batch already closed; no further writes"}
        else:
            self.closed = True
            if order_id != self.order.id or idempotency_key != self.order.id:
                result = {"status": "rejected", "posted": False, "reason": "Order or idempotency key mismatch"}
            elif type(amount_cents) is not int or amount_cents != self.order.merchandise_paid_cents:
                result = {
                    "status": "rejected",
                    "posted": False,
                    "reason": "REFUND_AMOUNT_POLICY_MISMATCH",
                    "submitted_cents": amount_cents,
                    "required_cents": self.order.merchandise_paid_cents,
                    "current_policy": (
                        "For authorized standard returns, refund merchandise_paid_cents only. "
                        "Outbound shipping is nonrefundable. invoice_total_cents includes shipping. "
                        "This supersedes the legacy invoice-total fallback."
                    ),
                }
            else:
                self.ledger.append(dict(request))
                result = {"status": "posted", "posted": True, **request}
        result["batch_closed"] = self.closed
        self.events.append({"operation": "submit", "attempted_at": attempted_at, "request": request, "result": result})
        return json.dumps(result)

    def verify(self, reply: str) -> Outcome:
        """Check business state, independently of model/judge claims or rule contents.

        This measures posting the correct refund, not the accuracy of final prose.
        Derive the expected ledger directly from the immutable order, rather than
        trusting the service's success flag or reusing its policy-check function.
        """
        expected = {
            "order_id": self.order.id,
            "amount_cents": self.order.merchandise_paid_cents,
            "idempotency_key": self.order.id,
        }
        submissions = [event for event in self.events if event["operation"] == "submit"]
        first_submit = next((i for i, event in enumerate(self.events) if event["operation"] == "submit"), 0)
        read_before_write = any(event.get("found") is True for event in self.events[:first_submit])
        checks = {
            "correct_ledger": self.ledger == [expected],
            "one_submission": len(submissions) == 1,
            "order_read_before_write": read_before_write,
            "nonempty_reply": bool(reply.strip()),
        }
        success = all(checks.values())
        return Outcome(
            scores={"task_completion": float(success)},
            success=success,
            evidence={"checks": checks, "expected_ledger": [expected], "actual_ledger": list(self.ledger)},
        )
