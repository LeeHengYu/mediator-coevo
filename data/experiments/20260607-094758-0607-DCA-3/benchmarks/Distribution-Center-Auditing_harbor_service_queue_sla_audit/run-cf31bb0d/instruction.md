# Task Instruction

Build two deliverables for an SLA audit from `/root/Ticket_Queue.xlsx`.

## Step 1: Inspect the source workbook
- Open `/root/Ticket_Queue.xlsx` with pandas/openpyxl.
- Read both sheets: `Tickets` and `SLA_Rules`.
- Print the columns and first few rows of each, and the full `SLA_Rules` table.
- Confirm `SLA_Rules` contains the columns needed: `Priority Tier`, `Max Open Hours`, `Escalation Required` (verify exact names; adapt if slightly different but do not rename in output).
- Confirm `Tickets` contains at least these columns: `Ticket ID`, `Queue`, `Priority Tier`, `Open Age Hours`, `Owner`, `Escalation Code`, `Region`, `Analyst`.

## Step 2: Create `/root/Service_Queue_SLA_Audit.xlsx`

Use openpyxl (or pandas ExcelWriter with openpyxl engine). Create exactly three sheets in this order: `RawData`, `Formatted Data`, `Summary`.

### Sheet `RawData`
- Write the `Tickets` table exactly as read (same columns, same order, same row order).

### Sheet `Formatted Data`
- Preserve the same row order as RawData.
- Columns 1-8 exactly (in this order, with these exact header strings):
  1. Ticket ID
  2. Queue
  3. Priority Tier
  4. Open Age Hours
  5. Owner
  6. Escalation Code
  7. Region
  8. Analyst
- Columns 9-12 exactly (these exact header strings):
  9. SLA Breach
  10. Missing Escalation
  11. Total Errors
  12. Error Summary
- Build a lookup dict from `SLA_Rules`: priority_tier -> (max_open_hours, escalation_required_flag).
- For each row, compute as concrete written values (not formulas):
  - `SLA Breach` = 1 if `Open Age Hours` > max_open_hours for that priority, else 0.
  - `Missing Escalation` = 1 if escalation_required == 'Y' for that priority AND `Escalation Code` is blank/NaN/empty-string, else 0.
  - `Total Errors` = sum of the two.
  - `Error Summary` exactly one of: `None`, `SLA Breach`, `Missing Escalation`, `SLA Breach, Missing Escalation` (use that exact comma+space joined form when both).
- Treat `Escalation Code` as blank when value is NaN, None, or an empty/whitespace-only string.
- Compare priority tiers as strings, stripped, to avoid lookup misses.

### Sheet `Summary`
- Headers exactly: `Queue`, `Region`, `SLA Breaches`, `Missing Escalations`, `Total Errors`.
- Aggregate from `Formatted Data` grouped by (`Queue`, `Region`): sum of SLA Breach, Missing Escalation, Total Errors.
- Keep only groups where `Total Errors` > 0.
- Sort by Queue ascending, then Region ascending.
- Append a final row: Queue=`Grand Total`, Region=`-`, then dataset-wide totals for SLA Breaches, Missing Escalations, Total Errors (sum across the filtered groups, which equals total across all rows since groups with 0 contribute nothing).

## Step 3: Create `/root/Service_Queue_SLA_Brief.docx`

Using python-docx, create a short executive summary (3-6 sentences) that includes:
- A plain-language definition of `SLA Breach` (ticket open longer than the priority's max hours) and `Missing Escalation` (priority requires escalation but escalation code is blank).
- The computed totals: SLA Breaches, Missing Escalations, Total Errors (use the Grand Total values).
- At least one actionable recommendation.
- Mention at least two high-priority queues (pick from Summary the queues with highest Total Errors, especially those associated with high-priority tiers) by name as having frequent exceptions.

Keep it concise (3-6 sentences). A simple paragraph or two is fine.

## Step 4: Validate before finishing
- Reopen `/root/Service_Queue_SLA_Audit.xlsx` and confirm:
  - Sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
  - `Formatted Data` has 12 columns with exact headers in order.
  - `Error Summary` values are only from the allowed set.
  - `Total Errors` = `SLA Breach` + `Missing Escalation` row-wise.
  - `Summary` includes only Total Errors > 0 rows, is sorted, and ends with a `Grand Total` / `-` row whose numbers equal column sums of the rows above.
- Reopen the .docx and confirm it contains both definitions, the three totals, a recommendation, and at least two queue names.

## Constraints
- Do not hardcode thresholds; read them from `SLA_Rules`.
- Keep filenames and sheet names exactly as specified.
- Write concrete values (not formulas) in the computed columns.

# Executor Policy

---
name: executor
description: Portable executor policy for workflow, verification, resource use, and failure handling across task runtimes.
---

## Executor Policy

Use this skill as execution policy, not as domain-specific task knowledge. When
task-local curated skills or resources are available, prefer them for domain
details and use this policy for workflow control.

## Task Execution

1. Read the task instruction, task resources, and verifier contract before editing.
2. Identify the scoring mechanism and the smallest command that can reproduce the
   failure or verify the expected behavior.
3. Inspect existing files and task-local resources before making changes.
4. Make the smallest source change that satisfies the task and verifier contract.
5. Keep a compact record of the concrete evidence behind the change: observed
   failure, files inspected, edit made, and verifier result.
6. Run targeted verification before broad verification when practical.

## File Editing

1. Read the actual current file contents immediately before making any edit.
   Never rely on memory, prior snapshots, or assumed content.
2. Prefer direct in-place edits over patch or diff application when the exact
   current context is uncertain.
3. If using a patch or diff, confirm that every context line exists verbatim in
   the file before applying it.
4. If a patch hunk fails to apply, re-read the affected file region and perform
   the edit directly instead of retrying the same patch.
5. After any edit, re-read the affected region to confirm the change landed.

## Build and Test Fixes

When a task requires fixing a broken build, failing test, or generated artifact:

1. Run the relevant build, test, or verifier command first to capture the
   baseline failure.
2. Identify the specific error message, file, line, or expected output before
   editing.
3. Apply the smallest fix, then re-run the same targeted command.
4. Treat newly introduced failures as separate sub-tasks and resolve them in
   order.
5. Do not mark the task complete until the verifier-relevant command succeeds or
   the remaining failure is clearly outside the task boundary.

## Artifact-Contract Handling

Do not treat artifacts as ordinary text files. Treat them as contract-bearing
interfaces between input data, generated output, verifier checks, and downstream
consumers.

When a task requires reading, modifying, or generating an artifact such as JSON,
DOT, reports, configs, generated source, schemas, datasets, or parsed outputs:

1. Identify the artifact contract first: format, schema, required fields,
   identifiers, references, ordering, examples, verifier assertions, and
   consuming code.
2. Inspect representative source artifacts directly before deciding how to
   transform or preserve them.
3. Determine whether the task calls for preservation, transformation, repair,
   generation, or validation.
4. Preserve required literals, identifiers, references, ordering, and
   representative content unless the contract explicitly requires a change.
5. Do not invent, drop, rename, normalize, collapse, expand, or repair artifact
   elements unless the verifier or consumer contract requires that behavior.
6. Prefer structured parsers, serializers, validators, or existing consumer code
   over ad hoc string manipulation when they are available.
7. After producing the artifact, run targeted checks for parseability, required
   keys or IDs, reference consistency, expected counts, preserved content, and
   format-specific validity.
8. If targeted checks regress or become unusable after a change, stop expanding
   the solution. Re-inspect the source contract and narrow the edit before trying
   a broader repair.

A plausible-looking artifact is not sufficient evidence. The artifact is only
correct when it satisfies the task contract under the verifier or consuming
code.

## Constraints

- Do not bypass, remove, or weaken tests, verifier scripts, fixtures, or expected
  output checks.
- Do not treat this policy as overriding task-specific instructions or verifier
  requirements.
- On tool or environment errors, retry once when the retry is safe, then report
  the failure with the command and error output.
- On ambiguous instructions, make a conservative assumption and continue.

# Task Resources

Inspect the task files, environment, tests, and expected outputs directly.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=hard, tags=[excel, openpyxl, docx, audit, service].
Verifier config: timeout_sec=900.0.