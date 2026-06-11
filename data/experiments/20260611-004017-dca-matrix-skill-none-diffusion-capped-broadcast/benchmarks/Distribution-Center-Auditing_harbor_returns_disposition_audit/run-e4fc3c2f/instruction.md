# Task Instruction

## Task: Distribution Center Returns Disposition Audit

You must produce two deliverable files:
1. `/root/Returns_Disposition_Audit.xlsx`
2. `/root/Returns_Disposition_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect the source files
- Read `/root/Return_Plan.xlsx` and print its sheet names, column headers, and first 5 rows.
- Read `/root/Disposition_Event_Log.xlsx` and print its sheet names, column headers, and first 5 rows. Pay close attention to the exact column names (e.g., `Event Status`, `Final Disposition`, `Return ID`, `Line ID`, and any timestamp/sequence column).
- Read `/root/Disposition_Alias.xlsx` and print its sheet names, column headers, and ALL rows (it is likely small). Note which column is the alias and which is the standard/canonical disposition.

Print all of this before writing any code that builds the output.

#### Step 1: Build the alias mapping
- From `Disposition_Alias.xlsx`, create a dictionary mapping each alias (lowercased) to its standard disposition (also lowercased for comparison, but store the original-case standard name for reference). If there is a column that is clearly the canonical/standard disposition and another that is the alias, use them accordingly. Print the mapping.

#### Step 2: Build the event lookup
- From `Disposition_Event_Log.xlsx`:
  - Filter to rows where `Event Status` == `COMPLETED` (case-insensitive match).
  - For each `(Return ID, Line ID)` group, keep only the latest row. Determine "latest" by whatever timestamp or sequence column exists (inspect the columns). If there is an event date/time column, sort by it descending and take the first. If there is a numeric event sequence/ID, use the max.
  - Store the resulting `Final Disposition` for each `(Return ID, Line ID)` pair.
  - Print how many unique (Return ID, Line ID) pairs have a completed event.

#### Step 3: Build `RawData` sheet
- Copy the plan table from `Return_Plan.xlsx` exactly (all columns, all rows, same order) into a DataFrame called `raw_data`.

#### Step 4: Build `Formatted Data` sheet
- Start from `raw_data`. Keep the same row order.
- The first 8 columns must be exactly: `Return ID`, `Line ID`, `Planned Disposition`, `Reason Code`, `Requested Qty`, `Warehouse`, `Carrier`, `Lane`. If the source columns have different names, rename them to match exactly. If the source has extra columns beyond these 8, drop them for the Formatted Data sheet.
- For each row, look up the `(Return ID, Line ID)` in the event lookup:
  - **Missing Final Event**: 1 if no COMPLETED event exists for that pair, else 0.
  - **Disposition Mismatch**: If a COMPLETED event exists, normalize the `Final Disposition` using the alias map (lowercase the final disposition, look it up in the alias dict; if found, use the mapped standard disposition; otherwise use the raw final disposition text). Compare (case-insensitive) to `Planned Disposition`. If they don't match, set to 1; else 0. If no COMPLETED event exists, set to 0.
  - **Total Errors** = Missing Final Event + Disposition Mismatch.
  - **Error Summary**: Exactly one of:
    - `None` (if Total Errors == 0)
    - `Missing Final Event` (if only that flag is 1)
    - `Disposition Mismatch` (if only that flag is 1)
    - `Missing Final Event, Disposition Mismatch` (if both are 1)
- Store these as concrete int/string values, NOT formulas.
- Print the value counts for `Missing Final Event`, `Disposition Mismatch`, and `Total Errors` to verify.
- Print any rows where `Total Errors > 0` so we can inspect them.

#### Step 5: Build `Summary` sheet
- From `Formatted Data`, group by `(Warehouse, Carrier)`.
- For each group, sum `Missing Final Event` → `Missing Final Events`, sum `Disposition Mismatch` → `Disposition Mismatches`, sum `Total Errors` → `Total Errors`.
- Keep only groups where `Total Errors > 0`.
- Sort by `Warehouse` ascending, then `Carrier` ascending.
- Append a Grand Total row: `Warehouse` = `Grand Total`, `Carrier` = `-`, and the remaining three columns = dataset-wide totals (sum across ALL rows in Formatted Data, not just the filtered groups — though they should be the same since zero-error groups contribute 0).
- The column headers must be exactly: `Warehouse`, `Carrier`, `Missing Final Events`, `Disposition Mismatches`, `Total Errors`.

#### Step 6: Write the Excel file
- Write `/root/Returns_Disposition_Audit.xlsx` with exactly three sheets named `RawData`, `Formatted Data`, `Summary` (in that order).
- Use `openpyxl` engine.
- Do NOT include the pandas index in any sheet.
- After writing, re-read the file and verify:
  - Sheet names are exactly `['RawData', 'Formatted Data', 'Summary']`.
  - `RawData` row count matches source.
  - `Formatted Data` has 12 columns with the correct headers.
  - `Summary` last row has `Warehouse` == `Grand Total`.
  - Print the Summary sheet contents.

#### Step 7: Write the Word document
- Create `/root/Returns_Disposition_Brief.docx` using `python-docx`.
- Write a short executive summary (3-6 sentences) that includes:
  1. A plain-language definition of both checks: "Missing Final Event" means no completed disposition event was recorded for a planned return line; "Disposition Mismatch" means the final recorded disposition differs from what was originally planned.
  2. The computed totals: state the exact numbers for Missing Final Events, Disposition Mismatches, and Total Errors from the Grand Total row.
  3. At least one actionable recommendation (e.g., investigate warehouse/carrier combinations with highest error rates, implement real-time disposition tracking alerts).
  4. Mention at least two specific Return IDs that have the most errors or appear most frequently in the error rows. Pick the Return IDs with the highest Total Errors (sum across their lines). Print which Return IDs you are selecting and why.

#### Step 8: Final verification
- Confirm both files exist at the correct paths.
- Re-read the Excel and print sheet names and row counts.
- Re-read the Word doc and print its text content.
- Print "TASK COMPLETE" when done.

### Important Notes
- Install any needed packages (`openpyxl`, `python-docx`) via pip if not already available.
- All column name matching with source data should be done carefully — print headers first, then map.
- Use concrete values (integers and strings), not Excel formulas.
- Keep filenames and sheet names EXACTLY as specified — they are case-sensitive.

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