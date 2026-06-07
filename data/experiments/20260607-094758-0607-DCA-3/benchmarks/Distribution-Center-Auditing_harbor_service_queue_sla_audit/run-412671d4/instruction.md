# Task Instruction

Execute the Service Queue SLA Audit task.

## Inputs
- Source workbook: `/root/Ticket_Queue.xlsx` with sheets `Tickets` and `SLA_Rules`.

## Step 1: Inspect the source
1. Load `/root/Ticket_Queue.xlsx` with pandas/openpyxl.
2. Print column names, dtypes, and a few rows for both `Tickets` and `SLA_Rules`.
3. Confirm the exact column names in `SLA_Rules` (expected: a Priority Tier column, `Max Open Hours`, `Escalation Required`). Use the actual names found.
4. Confirm the `Tickets` sheet has the 8 expected columns: Ticket ID, Queue, Priority Tier, Open Age Hours, Owner, Escalation Code, Region, Analyst.

## Step 2: Build `/root/Service_Queue_SLA_Audit.xlsx`

Create exactly three worksheets in this order: `RawData`, `Formatted Data`, `Summary`.

### RawData
- Copy the `Tickets` table exactly (same columns, same row order, same values).

### Formatted Data
- Preserve row order from `RawData`.
- Keep the first 8 columns exactly as listed in the task.
- Add columns 9-12: `SLA Breach`, `Missing Escalation`, `Total Errors`, `Error Summary`.
- Compute per row by looking up the row's Priority Tier in `SLA_Rules`:
  - `SLA Breach` = 1 if `Open Age Hours` > `Max Open Hours` for that tier, else 0.
  - `Missing Escalation` = 1 if `Escalation Required` == 'Y' for that tier AND `Escalation Code` is blank (NaN or empty/whitespace string), else 0.
  - `Total Errors` = `SLA Breach` + `Missing Escalation`.
  - `Error Summary`:
    - both 0 -> `None`
    - SLA only -> `SLA Breach`
    - Missing only -> `Missing Escalation`
    - both -> `SLA Breach, Missing Escalation`
- Write concrete integer/text values (no formulas).
- Treat blank Escalation Code as: NaN, None, empty string, or whitespace-only string.
- Compare `Escalation Required` case-insensitively against 'Y'.

### Summary
- Headers exactly: `Queue`, `Region`, `SLA Breaches`, `Missing Escalations`, `Total Errors`.
- Aggregate from `Formatted Data` grouped by (`Queue`, `Region`), summing `SLA Breach`, `Missing Escalation`, `Total Errors`.
- Drop any group with `Total Errors == 0`.
- Sort by `Queue` asc then `Region` asc.
- Append final row: `Queue`='Grand Total', `Region`='-', and column sums for the three numeric columns (sum over the included groups, which equals dataset totals because excluded groups contribute zero).

## Step 3: Build `/root/Service_Queue_SLA_Brief.docx`
Use python-docx. Write a 3-6 sentence executive summary that:
- Defines `SLA Breach` (ticket open longer than the Priority Tier's Max Open Hours threshold from SLA_Rules).
- Defines `Missing Escalation` (ticket whose Priority Tier requires escalation but Escalation Code is blank).
- States totals: SLA Breaches = X, Missing Escalations = Y, Total Errors = Z (use the computed numbers).
- Names at least two high-priority queues with the most exceptions (pick top queues by Total Errors among rows whose Priority Tier maps to the highest urgency, or simply the top-2 queues by Total Errors if priority info is not directly grouped; prefer queues that appear with high-priority tickets).
- Includes at least one actionable recommendation (e.g., reassign aging tickets, enforce escalation tagging).

## Step 4: Validate
1. Re-open `/root/Service_Queue_SLA_Audit.xlsx` and confirm sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
2. Verify `Formatted Data` has 12 columns with the exact headers in order.
3. Verify `Error Summary` values are only from the 4 allowed strings.
4. Verify `Summary` ends with the Grand Total row and column sums match the totals from `Formatted Data`.
5. Verify the docx exists and contains the totals and queue names.

## Constraints
- Do not hardcode priority thresholds; always look them up from `SLA_Rules`.
- Keep filenames and sheet names exact.
- Write literal values in the added columns, not formulas.

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