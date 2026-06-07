# Task Instruction

Execute a returns-processing disposition audit task. You must produce exactly two output files:

1. `/root/Returns_Disposition_Audit.xlsx`
2. `/root/Returns_Disposition_Brief.docx`

## Inputs (read first, do not modify)
- `/root/Return_Plan.xlsx` — plan table
- `/root/Disposition_Event_Log.xlsx` — event log with `Event Status` and `Final Disposition`
- `/root/Disposition_Alias.xlsx` — alias → standard disposition mapping

Before writing any output, open each input with pandas/openpyxl and inspect column names, dtypes, row counts, and a few sample rows. Confirm the exact column headers used for Return ID, Line ID, Event Status, Final Disposition, Event timestamp (or order column used for "latest"), alias column, and standard disposition column.

## Excel output: `/root/Returns_Disposition_Audit.xlsx`
Worksheets in this exact order and with these exact names: `RawData`, `Formatted Data`, `Summary`.

### Sheet 1 — `RawData`
- Copy the plan table from `Return_Plan.xlsx` exactly (same headers, same row order, same values, no added/removed columns).

### Sheet 2 — `Formatted Data`
- Same row order as `RawData`.
- First 8 columns must be exactly, in this order and with these exact header strings:
  1. `Return ID`
  2. `Line ID`
  3. `Planned Disposition`
  4. `Reason Code`
  5. `Requested Qty`
  6. `Warehouse`
  7. `Carrier`
  8. `Lane`
- Add columns 9–12 with exact headers:
  9. `Missing Final Event`
  10. `Disposition Mismatch`
  11. `Total Errors`
  12. `Error Summary`

Derivation rules:
- From `Disposition_Event_Log.xlsx`, filter to rows where `Event Status == 'COMPLETED'` (case as in the file; verify). For each `(Return ID, Line ID)`, keep only the latest such row. Determine "latest" using the timestamp/ordering column present in the log (inspect to confirm — likely an `Event Time`/`Timestamp` column; if multiple, use the most plausible chronological one). Ignore rows with any other status.
- Build alias map from `Disposition_Alias.xlsx`: lowercased alias → standard disposition. To normalize a kept event's `Final Disposition`: if its lowercased value matches an alias key, replace with the standard disposition; otherwise keep raw. Comparison to `Planned Disposition` is case-insensitive.
- For each plan row:
  - `Missing Final Event` = 1 if no kept COMPLETED event exists for that `(Return ID, Line ID)`, else 0.
  - `Disposition Mismatch` = 1 if a kept event exists AND normalized Final Disposition != Planned Disposition (case-insensitive); else 0. (If Missing Final Event = 1, Disposition Mismatch must be 0.)
  - `Total Errors` = sum of the two.
  - `Error Summary` must be exactly one of: `None`, `Missing Final Event`, `Disposition Mismatch`, `Missing Final Event, Disposition Mismatch`.
- Write concrete numeric/text values (integers 0/1, plain strings). Do NOT use spreadsheet formulas.

### Sheet 3 — `Summary`
- Exact headers in order: `Warehouse`, `Carrier`, `Missing Final Events`, `Disposition Mismatches`, `Total Errors`.
- Aggregate from `Formatted Data` grouped by `(Warehouse, Carrier)`, summing the three count columns.
- Include only groups where `Total Errors > 0`.
- Sort by `Warehouse` ascending, then `Carrier` ascending.
- Append a final Grand Total row: `Warehouse='Grand Total'`, `Carrier='-'`, remaining three columns = dataset-wide totals (sum across ALL Formatted Data rows, not just included groups — but since excluded groups have 0 errors, sums are equal; use full-dataset sum to be safe).

## Word output: `/root/Returns_Disposition_Brief.docx`
Use `python-docx`. Write a 3–6 sentence executive summary that includes:
- Plain-language definitions of both `Missing Final Event` (no completed disposition event was recorded for a planned return line) and `Disposition Mismatch` (a completed event exists but its normalized final disposition differs from the planned disposition).
- The computed totals: Missing Final Events, Disposition Mismatches, Total Errors (use the exact numbers from your Summary Grand Total).
- At least one concrete, actionable recommendation (e.g., follow up with specific warehouse/carrier pairs that have the most errors).
- Mention at least two high-priority Return IDs with the most exceptions (rank Return IDs by total errors across their lines and name the top 2 explicitly).

## Workflow
1. Inspect all three input files (print head, columns, dtypes, shape).
2. Build the kept-events map and alias map.
3. Construct the Formatted Data dataframe with all 12 columns and concrete values.
4. Build the Summary dataframe with the Grand Total row.
5. Write the workbook with all three sheets using a single ExcelWriter call so sheet order is preserved.
6. Compute top-2 Return IDs by total errors and totals for the brief; write the .docx.
7. Verify: reopen the xlsx and confirm sheet names, header rows, column counts, and that Error Summary values are in the allowed set. Reopen the .docx and confirm it has the required content.

## Constraints
- Filenames and worksheet names must match exactly (including spaces and capitalization).
- Do not add extra sheets or columns.
- Do not use formulas in the added columns.
- Preserve plan row order in both RawData and Formatted Data.

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