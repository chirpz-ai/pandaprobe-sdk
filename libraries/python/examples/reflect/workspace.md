# Reading a Reflect workspace

Start with `report.json` in the example's run folder. It summarizes task results,
proposals, validation, delivery, and cloud trace IDs. Reflect's durable records
are under `workspace/<identity-namespace>/`; the namespace separates platform
and project identities.

| File or folder | What it holds |
| --- | --- |
| `state.json` | Current accepted revision (`head`), proposals and decisions, evaluation progress, budgets, and session memory. |
| `revisions/<revision>/rules/**/*.md` | Immutable snapshots of rule text. Follow `state.json` → `head` to find the currently accepted rules; older snapshots preserve history. |
| `revisions/<revision>/manifest.json` | The rule filenames and content hashes for that snapshot. |
| `executions/*.json` | Each execution's working rules, active memory, trace IDs, and usage. Candidate edits are visible here before acceptance. |
| `evidence/*.txt` | Diagnostic evidence associated with failure notices. |
| `events/*.json` | The audit trail of proposals, validation decisions, and other lifecycle events. |
| `.lock` | Coordinates writes when multiple Reflect instances use the workspace. |

The failure example stores its own retrieval audit in `report.json`. The SDK
does not create delivery receipts or scan model inputs for rule text. Older run
artifacts may still contain historical receipt records; those files are preserved.

The agent sees a simpler filesystem through `reflect_bash`:

- `/rules/`: search and edit its working rules. Changes become proposals;
  independent validation publishes accepted changes into a new revision.
- `/context.md`: editable active notes for the session, separate from accepted rules.
- `/inbox/`: assigned failure notice and evidence, read-only.
- `/decisions/status.json`: recent validation outcomes, read-only.

These are isolated virtual paths, not folders at the root of your computer.
Their working contents are saved in the JSON records above, so you may not see
a standalone `context.md` on disk. Inspect the records freely; let Reflect manage
the revision hashes and state pointer instead of editing them by hand.
