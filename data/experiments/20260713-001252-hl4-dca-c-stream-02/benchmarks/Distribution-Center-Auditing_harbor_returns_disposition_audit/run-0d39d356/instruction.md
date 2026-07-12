# Task Instruction

## Task: Distribution Center Returns Disposition Audit

You must produce two output files:
1. `/root/Returns_Disposition_Audit.xlsx`
2. `/root/Returns_Disposition_Brief.docx`

### Step-by-step Instructions

#### Phase 1: Inspect Input Files

1. Read `/root/Return_Plan.xlsx` and print its sheet names, column headers, and first 5 rows.
2. Read `/root/Disposition_Event_Log.xlsx` and print its sheet names, column headers, and first 5 rows.
3. Read `/root/Disposition_Alias.xlsx` and print its sheet names, column headers, and first 5 rows.
4. Print the shape (row count, column count) of each file.
5. Print all unique values of the `Event Status` column in the event log.
6. Print all rows of the Disposition_Alias table (it is likely small).

#### Phase 2: Build the Audit Logic in Python

Use `openpyxl` for Excel writing and `python-docx` for Word. Install them if needed (`pip install openpyxl python-docx`).

##### 2a: Load data
- Load `Return_Plan.xlsx` into a DataFrame called `plan`. Preserve original column names and row order.
- Load `Disposition_Event_Log.xlsx` into a DataFrame called `events`.
- Load `Disposition_Alias.xlsx` into a DataFrame called `aliases`.

##### 2b: Build alias mapping
- From the alias table, build a dictionary mapping each alias (lowercased) to its standard disposition (lowercased). Print this dictionary to confirm correctness.

##### 2c: Determine the latest COMPLETED event per (Return ID, Line ID)
- Filter `events` to only rows where `Event Status` == `COMPLETED` (case-insensitive comparison to be safe).
- Among those, for each `(Return ID, Line ID)` group, keep only the row with the latest timestamp/sequence. Identify the timestamp or sequence column by inspecting the columns (likely `Event Timestamp` or similar). If there's a tie, keep the last row encountered.
- Store the result as `latest_events` — a DataFrame indexed or keyed by `(Return ID, Line ID)` with at least the `Final Disposition` column.

##### 2d: Normalize dispositions
- For each row in `latest_events`, take the `Final Disposition` value, lowercase it, look it up in the alias dictionary. If found, replace with the standard disposition. Store the normalized value.

##### 2e: Compute error columns
For each row in `plan` (maintaining original order):
- Look up `(Return ID, Line ID)` in `latest_events`.
- `Missing Final Event` = 1 if no matching COMPLETED event exists, else 0.
- `Disposition Mismatch` = 1 if a COMPLETED event exists AND the normalized Final Disposition (lowercased) != Planned Disposition (lowercased), else 0.
- `Total Errors` = Missing Final Event + Disposition Mismatch.
- `Error Summary`:
  - If Total Errors == 0: `"None"`
  - If Missing Final Event == 1 and Disposition Mismatch == 0: `"Missing Final Event"`
  - If Missing Final Event == 0 and Disposition Mismatch == 1: `"Disposition Mismatch"`
  - If both == 1: `"Missing Final Event, Disposition Mismatch"`

Print a summary of how many rows have each error type to verify.

#### Phase 3: Write `/root/Returns_Disposition_Audit.xlsx`

Use `openpyxl` to create the workbook with exactly three sheets in this order: `RawData`, `Formatted Data`, `Summary`. Remove any default sheets.

##### Sheet 1: `RawData`
- Copy the plan table exactly (all original columns, all rows, same order). Write headers in row 1, data starting row 2.

##### Sheet 2: `Formatted Data`
- Columns 1-8 must be exactly: `Return ID`, `Line ID`, `Planned Disposition`, `Reason Code`, `Requested Qty`, `Warehouse`, `Carrier`, `Lane`.
- Map from the plan DataFrame to these column names. If the plan uses slightly different names, map them carefully. Print the plan's original column names and confirm the mapping.
- Columns 9-12 headers exactly: `Missing Final Event`, `Disposition Mismatch`, `Total Errors`, `Error Summary`.
- Write concrete numeric values (0 or 1) for columns 9-11, and text strings for column 12. Do NOT write Excel formulas.
- Same row order as RawData.

##### Sheet 3: `Summary`
- Headers exactly: `Warehouse`, `Carrier`, `Missing Final Events`, `Disposition Mismatches`, `Total Errors`.
- Group the Formatted Data by (Warehouse, Carrier). Sum Missing Final Event → Missing Final Events, sum Disposition Mismatch → Disposition Mismatches, sum Total Errors → Total Errors.
- Include ONLY groups where Total Errors > 0.
- Sort by Warehouse ascending, then Carrier ascending.
- Append a final row: Warehouse=`Grand Total`, Carrier=`-`, and the dataset-wide totals for the three numeric columns.

#### Phase 4: Write `/root/Returns_Disposition_Brief.docx`

Create a Word document with a short executive summary (3-6 sentences) that includes:
- A plain-language definition of both checks: Missing Final Event means no completed disposition event was recorded for a return line; Disposition Mismatch means the final recorded disposition differs from what was planned.
- The computed totals for Missing Final Events, Disposition Mismatches, and Total Errors (use the Grand Total numbers).
- At least one actionable recommendation (e.g., investigate root causes, retrain staff, improve system alerts).
- Mention at least two specific Return IDs that have the most errors (find the Return IDs with the highest Total Errors sum across their lines and name them).

#### Phase 5: Validation

1. Re-read `/root/Returns_Disposition_Audit.xlsx` and verify:
   - Exactly 3 sheets named `RawData`, `Formatted Data`, `Summary`.
   - `RawData` row count matches the plan.
   - `Formatted Data` has 12 columns with exact header names as specified.
   - `Formatted Data` row count matches `RawData`.
   - `Summary` headers are exactly as specified.
   - `Summary` last row has Warehouse=`Grand Total`.
   - The Grand Total row's Total Errors equals the sum of Total Errors in Formatted Data.
2. Re-read `/root/Returns_Disposition_Brief.docx` and print its text to confirm it contains the required elements.
3. Print "VALIDATION PASSED" if all checks pass.

### Critical Constraints
- Output filenames must be exactly `/root/Returns_Disposition_Audit.xlsx` and `/root/Returns_Disposition_Brief.docx`.
- Sheet names must be exactly `RawData`, `Formatted Data`, `Summary`.
- Column headers must match exactly as specified (case-sensitive, spacing-sensitive).
- Error columns must contain concrete values, not formulas.
- Do NOT skip any validation step.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=hard, tags=[excel, openpyxl, docx, audit, returns].
Verifier config: timeout_sec=900.0.