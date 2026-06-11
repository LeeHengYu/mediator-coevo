# Task Instruction

Execute the following steps in a single Python script to produce `/root/Returns_Disposition_Audit.xlsx` and `/root/Returns_Disposition_Brief.docx`.

## Step 0 — Inspect source files

Before writing any processing code, read and print:
1. All column headers and first 5 rows of `/root/Return_Plan.xlsx`
2. All column headers and first 10 rows of `/root/Disposition_Event_Log.xlsx` (note every column name exactly, especially any column whose name contains "Event", "Status", "Disposition", "Return", "Line", or "Date/Time")
3. All column headers and all rows of `/root/Disposition_Alias.xlsx`

Print dtypes as well. This is critical — do NOT skip this step. Use the printed output to confirm exact column names before proceeding.

## Step 1 — Build RawData

Read `Return_Plan.xlsx` into a DataFrame `plan`. Write it unchanged to the `RawData` sheet of `/root/Returns_Disposition_Audit.xlsx`.

## Step 2 — Process Event Log

Read `Disposition_Event_Log.xlsx` into `events`.

- Filter to rows where `Event Status` == `COMPLETED` (use the exact column name found in Step 0; compare case-insensitively on the cell value, i.e., `.str.upper() == 'COMPLETED'`).
- Among the filtered rows, for each unique `(Return ID, Line ID)` group, keep only the row with the latest event timestamp. Identify the timestamp column from Step 0 (it may be called `Event Date`, `Event Timestamp`, `Timestamp`, or similar). Sort by that column descending within each group and take the first row.
- Store the result as `latest_events` with at minimum columns: `Return ID`, `Line ID`, `Final Disposition`.

## Step 3 — Build Alias Map

Read `Disposition_Alias.xlsx`. Build a Python dict mapping each alias (lowercased) to its standard disposition (lowercased). Print this dict for verification.

## Step 4 — Build Formatted Data

Start from `plan` (same row order). Keep the first 8 columns with exactly these headers:
  Return ID, Line ID, Planned Disposition, Reason Code, Requested Qty, Warehouse, Carrier, Lane

(If the source column names differ slightly, rename them to match exactly.)

Left-join `plan` with `latest_events` on `(Return ID, Line ID)`.

For each row:
- `Missing Final Event` = 1 if no matched COMPLETED event exists (i.e., `Final Disposition` is NaN after the join), else 0.
- To check disposition mismatch:
  - Take the `Final Disposition` from the joined event (not NaN).
  - Lowercase it. If it appears as a key in the alias dict, replace it with the alias dict's value.
  - Compare this normalized value to `Planned Disposition` (also lowercased).
  - `Disposition Mismatch` = 1 if they differ, else 0.
  - If `Missing Final Event` == 1, then `Disposition Mismatch` = 0 (no event to compare).
- `Total Errors` = `Missing Final Event` + `Disposition Mismatch`
- `Error Summary`:
  - If Total Errors == 0 → `None`
  - If Missing Final Event == 1 and Disposition Mismatch == 0 → `Missing Final Event`
  - If Missing Final Event == 0 and Disposition Mismatch == 1 → `Disposition Mismatch`
  - If both == 1 → `Missing Final Event, Disposition Mismatch`

Write concrete int values (not formulas) for the numeric columns and concrete strings for Error Summary.

Write this to the `Formatted Data` sheet.

## Step 5 — Build Summary

From the Formatted Data DataFrame:
- Group by `(Warehouse, Carrier)`.
- Sum `Missing Final Event` → `Missing Final Events`, `Disposition Mismatch` → `Disposition Mismatches`, `Total Errors` → `Total Errors`.
- Filter to groups where `Total Errors > 0`.
- Sort by Warehouse ascending, then Carrier ascending.
- Append a Grand Total row: Warehouse=`Grand Total`, Carrier=`-`, and sums of the three numeric columns across the FULL Formatted Data (not just the filtered groups — use the original DataFrame totals).
- Headers must be exactly: Warehouse, Carrier, Missing Final Events, Disposition Mismatches, Total Errors

Write to the `Summary` sheet.

## Step 6 — Build Word Document

Create `/root/Returns_Disposition_Brief.docx` with an executive summary (3–6 sentences) that includes:
- Plain-language definition of both checks: "Missing Final Event" means no completed disposition event was recorded; "Disposition Mismatch" means the final completed disposition differs from the planned disposition.
- The exact computed Grand Total numbers for Missing Final Events, Disposition Mismatches, and Total Errors (use the integer values from the Grand Total row).
- At least one actionable recommendation (e.g., investigate root causes, retrain staff, review carrier processes).
- Mention at least two specific Return IDs that have the most errors (find the Return IDs with the highest Total Errors sum across their lines, pick the top 2).

## Step 7 — Validation

After generating both files, re-read the Excel file and print:
- The `Formatted Data` sheet's added columns (columns 9-12) for every row.
- The full `Summary` sheet.
- Confirm sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
- Print the Word document's full text.

Use `pandas` and `openpyxl` for Excel, `python-docx` for Word. Install any missing packages with pip first.

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