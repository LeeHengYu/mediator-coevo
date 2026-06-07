# Task Instruction

Build two deliverables from `/root/Ticket_Queue.xlsx`: an Excel audit workbook at `/root/Service_Queue_SLA_Audit.xlsx` and a Word brief at `/root/Service_Queue_SLA_Brief.docx`.

## Step 1: Inspect the source workbook
- Open `/root/Ticket_Queue.xlsx` and read both sheets: `Tickets` and `SLA_Rules`.
- Print column names, dtypes, row counts, and a sample of values for each sheet.
- Specifically check whether the `Tickets` sheet contains literal string values like `N/A`, blanks, or other sentinels in any column (especially `Escalation Code`). Read the workbook in a way that preserves these literals (e.g., `pd.read_excel(..., keep_default_na=False, na_values=[])` or by reading cells directly via openpyxl). Do NOT let pandas coerce `N/A`, `NA`, `NULL`, or similar literal strings into NaN/None.
- Inspect `SLA_Rules` to confirm its columns. Expected columns include `Priority Tier`, `Max Open Hours`, and `Escalation Required` (values like `Y`/`N`). Use whatever exact column names exist.

## Step 2: Build `/root/Service_Queue_SLA_Audit.xlsx`

Create exactly three sheets in this order: `RawData`, `Formatted Data`, `Summary`.

### Sheet `RawData`
- Write the `Tickets` table EXACTLY as read from source, preserving all string literals (including any `N/A`, empty strings, whitespace). Preserve column order, headers, row order, and cell values verbatim.
- After writing, re-open the produced file and confirm a sampled cell that originally held `N/A` still reads back as the string `N/A` (not None/NaN/empty).

### Sheet `Formatted Data`
- Same row order as `RawData`.
- First 8 columns exactly (headers verbatim): `Ticket ID`, `Queue`, `Priority Tier`, `Open Age Hours`, `Owner`, `Escalation Code`, `Region`, `Analyst`.
- Add columns 9-12 with headers exactly: `SLA Breach`, `Missing Escalation`, `Total Errors`, `Error Summary`.
- Build a lookup from `SLA_Rules` keyed by `Priority Tier` -> (`Max Open Hours`, `Escalation Required`). Do not hardcode thresholds.
- For each row:
  - `SLA Breach` = 1 if `Open Age Hours` > `Max Open Hours` for that row's `Priority Tier`, else 0.
  - `Missing Escalation` = 1 if `Escalation Required` == `Y` for that priority AND `Escalation Code` is blank (treat truly empty/whitespace as blank; do NOT treat literal `N/A` as blank unless the source actually uses `N/A` to mean missing — follow the literal: if source has empty cell, blank; if it has `N/A`, that is a non-blank string and is NOT a missing escalation). If unsure, base blankness on `value is None or str(value).strip() == ''`.
  - `Total Errors` = `SLA Breach + Missing Escalation`.
  - `Error Summary` exactly one of: `None`, `SLA Breach`, `Missing Escalation`, `SLA Breach, Missing Escalation`.
- Write concrete numeric/text values, not formulas.

### Sheet `Summary`
- Headers exactly: `Queue`, `Region`, `SLA Breaches`, `Missing Escalations`, `Total Errors`.
- Aggregate from `Formatted Data` grouped by `(Queue, Region)`; include only groups with `Total Errors > 0`.
- Sort by `Queue` ascending, then `Region` ascending.
- Append a final row: `Queue` = `Grand Total`, `Region` = `-`, remaining columns = dataset totals (sum across the included groups equals dataset totals; confirm this matches sums over all of `Formatted Data`).

## Step 3: Build `/root/Service_Queue_SLA_Brief.docx`
- 3-6 sentence executive summary including:
  - Plain-language definition of `SLA Breach` (ticket open longer than the priority's max open hours) and `Missing Escalation` (priority requires escalation but escalation code is blank).
  - The computed totals for SLA Breaches, Missing Escalations, and Total Errors (from the Grand Total row).
  - At least one actionable recommendation.
  - Mention at least two high-priority queues with frequent exceptions (pick top queues by total errors, ideally those tied to high-priority tiers).

## Step 4: Validate before finishing
- Reopen `/root/Service_Queue_SLA_Audit.xlsx` and verify:
  - Sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
  - `RawData` matches `Tickets` cell-for-cell, including literal strings like `N/A` (spot-check several rows).
  - `Formatted Data` has 12 columns with exact headers in the specified order.
  - `Error Summary` values are one of the 4 allowed strings and consistent with the 0/1 flags.
  - `Summary` headers, sort order, filter (Total Errors > 0), and `Grand Total` row are correct, and the Grand Total equals sum of all `Formatted Data` errors.
- Confirm `/root/Service_Queue_SLA_Brief.docx` exists and contains the required content.

## Critical constraints
- Preserve source literals exactly in `RawData` (do not coerce `N/A` to None/NaN). Use `keep_default_na=False, na_values=[]` when reading with pandas, or copy via openpyxl cell-by-cell.
- Use thresholds from `SLA_Rules`; do not hardcode by priority tier.
- Keep filenames and sheet names exactly as specified.

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