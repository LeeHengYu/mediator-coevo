# Task Instruction

## Task: Service Queue SLA Audit

You need to read `/root/Ticket_Queue.xlsx`, process the data, and produce two output files:
1. `/root/Service_Queue_SLA_Audit.xlsx`
2. `/root/Service_Queue_SLA_Brief.docx`

### Step 0: Inspect the Source Workbook

Read `/root/Ticket_Queue.xlsx` and inspect both sheets:
- `Tickets` sheet: print the column headers, first 5 rows, and total row count.
- `SLA_Rules` sheet: print all rows (it should be small). Note the columns — expect at least `Priority Tier`, `Max Open Hours`, and `Escalation Required`.

Confirm exact column names before proceeding. Do NOT assume column names.

### Step 1: Build the Output Excel Workbook

Use Python with `openpyxl` and `pandas`. Create `/root/Service_Queue_SLA_Audit.xlsx` with exactly three worksheets: `RawData`, `Formatted Data`, `Summary`.

#### 1a) `RawData` Sheet
- Copy the entire `Tickets` table exactly as-is (headers + all rows, same order).

#### 1b) `Formatted Data` Sheet
- Same row order as `RawData`.
- First 8 columns with exactly these headers (in this order):
  1. `Ticket ID`
  2. `Queue`
  3. `Priority Tier`
  4. `Open Age Hours`
  5. `Owner`
  6. `Escalation Code`
  7. `Region`
  8. `Analyst`
- Map source columns to these headers. The source may use slightly different names — map by inspecting the actual headers from Step 0. For example, if the source has `Ticket_ID`, map it to `Ticket ID`.
- Add 4 new columns (columns 9–12) with exactly these headers:
  9. `SLA Breach`
  10. `Missing Escalation`
  11. `Total Errors`
  12. `Error Summary`

**Computation rules** (use the `SLA_Rules` sheet data — do NOT hardcode thresholds):
- Build a lookup dict from `SLA_Rules`: for each `Priority Tier`, store `Max Open Hours` (numeric) and `Escalation Required` (string, 'Y' or 'N').
- `SLA Breach` = 1 if the row's `Open Age Hours` > the `Max Open Hours` for that row's `Priority Tier`; else 0.
- `Missing Escalation` = 1 if `Escalation Required` is `'Y'` for that row's `Priority Tier` AND the row's `Escalation Code` is blank/empty/NaN; else 0.
- `Total Errors` = `SLA Breach` + `Missing Escalation`.
- `Error Summary`:
  - If both are 0: `None`
  - If only SLA Breach is 1: `SLA Breach`
  - If only Missing Escalation is 1: `Missing Escalation`
  - If both are 1: `SLA Breach, Missing Escalation`

**IMPORTANT**: Write concrete values (integers and strings), not Excel formulas.

#### 1c) `Summary` Sheet
- Headers (exactly): `Queue`, `Region`, `SLA Breaches`, `Missing Escalations`, `Total Errors`
- Group the `Formatted Data` by `(Queue, Region)` and sum `SLA Breach`, `Missing Escalation`, `Total Errors` for each group.
- Include ONLY groups where `Total Errors > 0`.
- Sort by `Queue` ascending, then `Region` ascending.
- Append a final row: `Queue` = `Grand Total`, `Region` = `-`, and the remaining 3 columns = dataset-wide totals (sum of ALL rows from Formatted Data, not just the filtered groups — but since we only include groups with errors > 0 and the grand total should reflect the entire dataset, sum from the full Formatted Data).

### Step 2: Build the Word Document

Create `/root/Service_Queue_SLA_Brief.docx` using `python-docx`. Write an executive summary paragraph (3–6 sentences) that includes:
- A plain-language definition of both checks: what `SLA Breach` means (ticket open longer than the allowed max hours for its priority tier) and what `Missing Escalation` means (escalation required by policy but no escalation code recorded).
- The computed totals: total SLA Breaches, total Missing Escalations, total Total Errors (use the actual numbers from your computation).
- Mention at least two specific queues that have the highest error counts (look at your Summary data to identify them).
- At least one actionable recommendation (e.g., implement automated escalation alerts, review staffing for high-breach queues).

### Step 3: Validate

After creating both files:
1. Re-read `/root/Service_Queue_SLA_Audit.xlsx` and verify:
   - Sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
   - `RawData` row count matches source `Tickets` row count.
   - `Formatted Data` has 12 columns with exact header names listed above.
   - `Formatted Data` row count matches `RawData`.
   - `Summary` last row has `Queue` = `Grand Total` and `Region` = `-`.
   - `Summary` is sorted by Queue then Region ascending.
   - Grand Total numbers match the sums from Formatted Data.
2. Re-read `/root/Service_Queue_SLA_Brief.docx` and print its text content to confirm it contains the required elements.
3. Print confirmation of all checks passing.

### Important Notes
- Use `SLA_Rules` data dynamically — do not hardcode priority tier thresholds.
- Be careful with blank/NaN handling for `Escalation Code`: treat NaN, None, empty string, and whitespace-only strings all as "blank".
- Ensure filenames and sheet names are character-perfect.
- When writing to Excel with openpyxl, make sure `Open Age Hours` values remain numeric (not strings).
- For the `Formatted Data` sheet, if source column names differ from the required output names, rename them explicitly.

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