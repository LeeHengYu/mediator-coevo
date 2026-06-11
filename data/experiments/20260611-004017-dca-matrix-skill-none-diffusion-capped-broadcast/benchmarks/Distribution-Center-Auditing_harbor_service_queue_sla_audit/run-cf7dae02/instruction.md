# Task Instruction

## Task: Service Queue SLA Audit

You must read `/root/Ticket_Queue.xlsx` and produce two output files:
1. `/root/Service_Queue_SLA_Audit.xlsx`
2. `/root/Service_Queue_SLA_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect the source workbook
- Open `/root/Ticket_Queue.xlsx` and list the sheet names.
- Read the `Tickets` sheet fully. Print the column headers and first 5 rows. Note the total row count.
- Read the `SLA_Rules` sheet fully. Print all rows. Note the columns — expect at least `Priority Tier`, `Max Open Hours`, and `Escalation Required`.
- Identify the exact column names used in both sheets (they may differ slightly from the instruction's labels). Map them carefully.

#### Step 1: Build `RawData` sheet
- Copy the entire `Tickets` table as-is into a sheet named `RawData` in the output workbook.

#### Step 2: Build `Formatted Data` sheet
- Start with the same rows in the same order as `RawData`.
- Keep the first 8 columns with exactly these headers (rename if needed):
  1. `Ticket ID`
  2. `Queue`
  3. `Priority Tier`
  4. `Open Age Hours`
  5. `Owner`
  6. `Escalation Code`
  7. `Region`
  8. `Analyst`
- Create a lookup dictionary from `SLA_Rules`: for each `Priority Tier`, store `Max Open Hours` (numeric) and `Escalation Required` (string, 'Y' or 'N').
- For each row, compute:
  - `SLA Breach`: 1 if `Open Age Hours` > `Max Open Hours` for that row's `Priority Tier`, else 0. (Use strict greater-than.)
  - `Missing Escalation`: 1 if the SLA rule says `Escalation Required` == 'Y' for that priority tier AND the row's `Escalation Code` is blank/empty/NaN, else 0.
  - `Total Errors`: `SLA Breach` + `Missing Escalation` (integer).
  - `Error Summary`: Construct from the flags:
    - Both 0 → `None`
    - Only SLA Breach → `SLA Breach`
    - Only Missing Escalation → `Missing Escalation`
    - Both 1 → `SLA Breach, Missing Escalation`
- Write these as concrete integer/string values (not Excel formulas).
- The sheet must have exactly 12 columns with the exact headers listed above.

#### Step 3: Build `Summary` sheet
- From `Formatted Data`, group by `(Queue, Region)` and sum `SLA Breach`, `Missing Escalation`, `Total Errors`.
- Filter to only groups where `Total Errors > 0`.
- Sort by `Queue` ascending, then `Region` ascending.
- Headers must be exactly: `Queue`, `Region`, `SLA Breaches`, `Missing Escalations`, `Total Errors`.
- Append a final row: `Queue` = `Grand Total`, `Region` = `-`, and the remaining columns are the dataset-wide totals of SLA Breaches, Missing Escalations, and Total Errors (summed across ALL rows in Formatted Data, not just the filtered groups — though they should be the same since groups with 0 errors contribute 0).
- Write all values as concrete numbers.

#### Step 4: Save the Excel file
- Save as `/root/Service_Queue_SLA_Audit.xlsx` with exactly three sheets: `RawData`, `Formatted Data`, `Summary` (in that order).
- Use `openpyxl` as the engine.

#### Step 5: Create the Word document
- Create `/root/Service_Queue_SLA_Brief.docx` using `python-docx`.
- Write an executive summary paragraph (3-6 sentences) that includes:
  - A plain-language definition of both checks: SLA Breach (ticket open longer than the allowed max hours for its priority tier) and Missing Escalation (ticket's priority tier requires escalation but no escalation code was recorded).
  - The computed totals: state the exact numbers for SLA Breaches, Missing Escalations, and Total Errors from the Grand Total row.
  - At least one actionable recommendation (e.g., "prioritize resolving aged P1 tickets and ensure escalation codes are assigned").
  - Mention at least two specific queues that have the highest error counts (look at the Summary data to identify them).

#### Step 6: Validate
- Re-read `/root/Service_Queue_SLA_Audit.xlsx` and verify:
  - Sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
  - `RawData` row count matches the source `Tickets` sheet.
  - `Formatted Data` has 12 columns with the exact specified headers.
  - `Summary` last row has `Queue` == `Grand Total` and `Region` == `-`.
  - `Summary` Grand Total `Total Errors` == sum of `SLA Breaches` + `Missing Escalations` in that row.
  - Print the Summary table to confirm correctness.
- Re-read `/root/Service_Queue_SLA_Brief.docx` and print its text to confirm it contains the required elements.

### Important Notes
- Do NOT hardcode SLA thresholds. Read them from the `SLA_Rules` sheet.
- Be careful with blank/NaN detection for `Escalation Code`. Use `pd.isna()` or check for empty string after converting.
- Ensure `Open Age Hours` comparison is numeric (cast if needed).
- Column name mapping: the source `Tickets` sheet may already use the exact names, or may use slightly different names. Inspect first, then map.
- If `python-docx` is not installed, install it with `pip install python-docx`.
- If `openpyxl` is not installed, install it with `pip install openpyxl`.

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