# Task Instruction

## Task: Service Queue SLA Audit

You must read `/root/Ticket_Queue.xlsx` and produce two deliverables:
1. `/root/Service_Queue_SLA_Audit.xlsx`
2. `/root/Service_Queue_SLA_Brief.docx`

### Step-by-step instructions

#### Step 1: Inspect the source workbook
- Open `/root/Ticket_Queue.xlsx` using openpyxl or pandas.
- Read the `Tickets` sheet — note all column names and their order. Print the first 5 rows and all column names.
- Read the `SLA_Rules` sheet — note all column names. Print all rows. You need at minimum: `Priority Tier`, `Max Open Hours`, and `Escalation Required` columns (names may vary slightly — inspect carefully).
- Print the exact column names from both sheets so you can map them correctly.

#### Step 2: Build the SLA lookup
- From `SLA_Rules`, create a dictionary keyed by `Priority Tier` with values for `Max Open Hours` (numeric) and `Escalation Required` (string, likely 'Y' or 'N').
- Print this dictionary to confirm correctness.

#### Step 3: Create the output Excel workbook
Use openpyxl to create `/root/Service_Queue_SLA_Audit.xlsx` with exactly three sheets named: `RawData`, `Formatted Data`, `Summary`.

##### Sheet: `RawData`
- Copy the entire `Tickets` table exactly as-is (headers + all data rows, same order).

##### Sheet: `Formatted Data`
- Same row order as RawData.
- The first 8 columns must be exactly these headers (use these exact strings regardless of source column names):
  1. `Ticket ID`
  2. `Queue`
  3. `Priority Tier`
  4. `Open Age Hours`
  5. `Owner`
  6. `Escalation Code`
  7. `Region`
  8. `Analyst`
- Map source columns to these 8 columns. The source may have columns named identically or very similarly — inspect and map carefully. `Escalation Code` in the source might be the column that holds the actual escalation code value (could be blank/None for some rows).
- Add 4 new columns (9-12) with exactly these headers:
  9. `SLA Breach`
  10. `Missing Escalation`
  11. `Total Errors`
  12. `Error Summary`
- Compute for each row:
  - `SLA Breach`: 1 if the row's `Open Age Hours` > the `Max Open Hours` for that row's `Priority Tier` (from SLA_Rules lookup), else 0.
  - `Missing Escalation`: 1 if `Escalation Required` is `Y` for that row's `Priority Tier` AND the row's `Escalation Code` is blank/None/empty string, else 0.
  - `Total Errors`: `SLA Breach + Missing Escalation` (integer).
  - `Error Summary`: exactly one of these strings:
    - `"None"` if Total Errors == 0
    - `"SLA Breach"` if SLA Breach == 1 and Missing Escalation == 0
    - `"Missing Escalation"` if SLA Breach == 0 and Missing Escalation == 1
    - `"SLA Breach, Missing Escalation"` if both == 1
- Write concrete values (integers and strings), NOT formulas.

##### Sheet: `Summary`
- Headers (exactly): `Queue`, `Region`, `SLA Breaches`, `Missing Escalations`, `Total Errors`
- Group the `Formatted Data` rows by `(Queue, Region)`.
- For each group, sum `SLA Breach` → `SLA Breaches`, sum `Missing Escalation` → `Missing Escalations`, sum `Total Errors` → `Total Errors`.
- Include ONLY groups where `Total Errors > 0`.
- Sort by `Queue` ascending then `Region` ascending (standard alphabetical).
- Append a final row: `Queue` = `Grand Total`, `Region` = `-`, and the remaining three columns = the dataset-wide totals (sum across ALL rows in Formatted Data, not just the filtered groups — but since groups with 0 errors are excluded and their totals are 0, summing the included groups gives the same result; still, compute from the full Formatted Data to be safe).

#### Step 4: Create the Word document
- Use python-docx to create `/root/Service_Queue_SLA_Brief.docx`.
- Write a short executive summary (3-6 sentences) that includes:
  - A plain-language definition of both checks: SLA Breach (ticket open longer than the allowed max hours for its priority tier) and Missing Escalation (ticket's priority tier requires escalation but no escalation code is recorded).
  - The computed grand totals for SLA Breaches, Missing Escalations, and Total Errors (use the actual numbers from your computation).
  - At least one actionable recommendation (e.g., implement automated escalation alerts, review staffing for high-breach queues).
  - Mention at least two specific queues that have the highest error counts (look at the Summary sheet data to identify these).

#### Step 5: Validate
- Re-open `/root/Service_Queue_SLA_Audit.xlsx` and verify:
  - Sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
  - `RawData` row count matches source `Tickets` row count.
  - `Formatted Data` has 12 columns with the exact header names specified.
  - `Formatted Data` row count matches `RawData`.
  - `Summary` has 5 columns with exact header names. Last row has Queue = `Grand Total`.
  - Grand Total row totals match the sum of SLA Breach and Missing Escalation columns from Formatted Data.
- Re-open `/root/Service_Queue_SLA_Brief.docx` and print its text to confirm it has the required content.
- Print confirmation of all checks passing.

### Important notes
- Do NOT hardcode SLA thresholds — always read them from the `SLA_Rules` sheet.
- Use exact filenames and sheet names as specified.
- If any column name in the source doesn't exactly match what's expected, print all column names and map them carefully before proceeding.
- Install any needed packages (openpyxl, python-docx) if not already available.

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