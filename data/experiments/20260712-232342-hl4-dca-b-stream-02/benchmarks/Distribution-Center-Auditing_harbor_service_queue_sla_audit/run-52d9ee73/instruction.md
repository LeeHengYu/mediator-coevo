# Task Instruction

## Task: Service Queue SLA Audit

You need to read `/root/Ticket_Queue.xlsx` and produce two output files:
1. `/root/Service_Queue_SLA_Audit.xlsx`
2. `/root/Service_Queue_SLA_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect the source workbook
- Open `/root/Ticket_Queue.xlsx` and examine both sheets: `Tickets` and `SLA_Rules`.
- Print the column headers and first few rows of each sheet.
- Print all rows of `SLA_Rules` so you know the exact Priority Tier names, `Max Open Hours` values, and `Escalation Required` values.
- Pay close attention to the exact column names in both sheets (they may differ slightly from the task description).
- Note: `Escalation Code` being "blank" means it is NaN/None/empty string. Check for all these variants.

#### Step 1: Build `RawData` sheet
- Copy the `Tickets` sheet exactly (all columns, all rows, same order) into a sheet named `RawData` in the output workbook.

#### Step 2: Build `Formatted Data` sheet
- Start with the same rows in the same order as `RawData`.
- Keep only the first 8 columns with EXACTLY these headers (rename if needed):
  1. `Ticket ID`
  2. `Queue`
  3. `Priority Tier`
  4. `Open Age Hours`
  5. `Owner`
  6. `Escalation Code`
  7. `Region`
  8. `Analyst`
- Merge/lookup from `SLA_Rules` on `Priority Tier` to get `Max Open Hours` and `Escalation Required` for each row.
- Compute four new columns (as concrete values, NOT formulas):
  - **Column 9 `SLA Breach`**: 1 if `Open Age Hours` > `Max Open Hours` for that row's Priority Tier, else 0. Use strict greater-than (not >=).
  - **Column 10 `Missing Escalation`**: 1 if `Escalation Required` == 'Y' for that row's Priority Tier AND `Escalation Code` is blank/empty/NaN, else 0.
  - **Column 11 `Total Errors`**: `SLA Breach` + `Missing Escalation` (integer).
  - **Column 12 `Error Summary`**: Exactly one of these strings:
    - `None` (if Total Errors == 0)
    - `SLA Breach` (if only SLA Breach == 1)
    - `Missing Escalation` (if only Missing Escalation == 1)
    - `SLA Breach, Missing Escalation` (if both == 1)
- IMPORTANT: Write these as plain integer/string values, not Excel formulas.
- Double-check: print a few rows where you expect breaches and a few where you don't, to verify the logic is correct.

#### Step 3: Build `Summary` sheet
- Aggregate from `Formatted Data` by `(Queue, Region)` groups.
- Columns with EXACTLY these headers:
  1. `Queue`
  2. `Region`
  3. `SLA Breaches` (sum of `SLA Breach` column for that group)
  4. `Missing Escalations` (sum of `Missing Escalation` column for that group)
  5. `Total Errors` (sum of `Total Errors` column for that group)
- Include ONLY groups where `Total Errors > 0`.
- Sort by `Queue` ascending, then `Region` ascending.
- Append a final row: `Queue` = `Grand Total`, `Region` = `-`, and the remaining columns = dataset-wide totals (sum of ALL rows from Formatted Data, not just the filtered groups — but since groups with 0 errors are excluded and their totals are 0, the sums should be the same).

#### Step 4: Save the Excel file
- Save as `/root/Service_Queue_SLA_Audit.xlsx` with exactly three sheets: `RawData`, `Formatted Data`, `Summary`.
- Use openpyxl engine.

#### Step 5: Build the Word document
- Create `/root/Service_Queue_SLA_Brief.docx` with a short executive summary (3-6 sentences).
- Must include:
  - A plain-language definition of both checks: what `SLA Breach` means (ticket open longer than the allowed max hours for its priority tier) and what `Missing Escalation` means (ticket's priority tier requires escalation but no escalation code is recorded).
  - The computed grand totals for SLA Breaches, Missing Escalations, and Total Errors (use the exact numbers from your Summary Grand Total row).
  - At least one actionable recommendation.
  - Mention at least two specific queues that have frequent/high error counts (pick the top 2 queues by Total Errors from your Summary data).
- Save as `/root/Service_Queue_SLA_Brief.docx`.

#### Step 6: Verification
- Re-open `/root/Service_Queue_SLA_Audit.xlsx` and verify:
  - Sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
  - `RawData` row count matches source `Tickets` sheet.
  - `Formatted Data` has 12 columns with exact headers listed above.
  - `Summary` has 5 columns with exact headers, only error groups, sorted correctly, with Grand Total row at end.
  - Spot-check a few SLA Breach and Missing Escalation values against the SLA_Rules thresholds.
- Re-open `/root/Service_Queue_SLA_Brief.docx` and print its text to verify it contains the required elements.

### Key Pitfalls to Avoid (from prior feedback)
- Do NOT hardcode SLA thresholds — always read them from `SLA_Rules`.
- Make sure blank/NaN escalation codes are properly detected (check for `pd.isna()`, empty string `''`, and whitespace-only strings).
- Ensure the Grand Total row uses the correct computed totals.
- Ensure all numbers in the Word doc match the Excel Summary exactly.
- Column header names must be letter-perfect (e.g., `SLA Breaches` in Summary, `SLA Breach` in Formatted Data).

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