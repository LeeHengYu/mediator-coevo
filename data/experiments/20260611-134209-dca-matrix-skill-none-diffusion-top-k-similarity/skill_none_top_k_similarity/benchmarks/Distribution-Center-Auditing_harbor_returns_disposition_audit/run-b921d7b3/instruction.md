# Task Instruction

## Task: Distribution Center Returns Disposition Audit

You must produce two files:
1. `/root/Returns_Disposition_Audit.xlsx`
2. `/root/Returns_Disposition_Brief.docx`

### Step-by-step instructions:

#### Step 0: Inspect the input files
- Read `/root/Return_Plan.xlsx` and print its sheet names, columns, first 10 rows, and total row count.
- Read `/root/Disposition_Event_Log.xlsx` and print its sheet names, columns, first 10 rows, and total row count.
- Read `/root/Disposition_Alias.xlsx` and print its sheet names, columns, first 10 rows, and total row count.
- Identify the exact column names in each file (they may differ slightly from what's described). Print them all.

#### Step 1: Build the `RawData` sheet
- Copy the entire plan table from `Return_Plan.xlsx` as-is into a DataFrame called `raw_data`. This will become the `RawData` worksheet.

#### Step 2: Build the `Formatted Data` sheet

**2a: Prepare the base columns**
- Start with `raw_data` (same row order). The first 8 columns must be exactly:
  1. Return ID
  2. Line ID
  3. Planned Disposition
  4. Reason Code
  5. Requested Qty
  6. Warehouse
  7. Carrier
  8. Lane
- If the source columns have different names, rename them to match exactly. Print the mapping.

**2b: Process the Event Log**
- From `Disposition_Event_Log.xlsx`, filter to only rows where `Event Status` == `COMPLETED` (case-insensitive comparison on Event Status).
- For each `(Return ID, Line ID)` group, keep only the row with the latest event. If there's a timestamp/date column, use that for ordering. If there's a sequence or row-order column, use that. Print what column you used for ordering and show a few examples.
- Store this as `completed_events` — a lookup from `(Return ID, Line ID)` to `Final Disposition`.

**2c: Build the Disposition Alias mapping**
- From `Disposition_Alias.xlsx`, build a dictionary mapping each alias (lowercased) to its standard disposition (lowercased).
- Print this mapping.

**2d: Compute the 4 new columns**
For each row in `Formatted Data`:
- Look up `(Return ID, Line ID)` in `completed_events`.
- `Missing Final Event` = 1 if no completed event exists, else 0.
- If a completed event exists:
  - Get the `Final Disposition` value.
  - Normalize it: if `final_disposition.strip().lower()` is a key in the alias dict, replace it with the standard disposition.
  - Compare normalized final disposition (lowercased) to `Planned Disposition` (lowercased, stripped).
  - `Disposition Mismatch` = 1 if they don't match, else 0.
- If no completed event: `Disposition Mismatch` = 0.
- `Total Errors` = `Missing Final Event` + `Disposition Mismatch`.
- `Error Summary`:
  - If both flags are 0: `None`
  - If only Missing Final Event == 1: `Missing Final Event`
  - If only Disposition Mismatch == 1: `Disposition Mismatch`
  - If both == 1: `Missing Final Event, Disposition Mismatch`

Print the value counts for each of the 4 new columns to verify.

#### Step 3: Build the `Summary` sheet
- From `Formatted Data`, group by `(Warehouse, Carrier)`.
- For each group, sum `Missing Final Event`, `Disposition Mismatch`, and `Total Errors`.
- Filter to only groups where `Total Errors > 0`.
- Sort by `Warehouse` ascending, then `Carrier` ascending.
- Append a Grand Total row: Warehouse=`Grand Total`, Carrier=`-`, and the sums of the three numeric columns across the entire Formatted Data (not just filtered groups — use all rows).
- The columns must be exactly: `Warehouse`, `Carrier`, `Missing Final Events`, `Disposition Mismatches`, `Total Errors`.
- Print the summary table.

#### Step 4: Write the Excel file
- Use `openpyxl` via pandas `ExcelWriter` to write `/root/Returns_Disposition_Audit.xlsx` with exactly three sheets: `RawData`, `Formatted Data`, `Summary`.
- For the `Formatted Data` sheet, write concrete values (not formulas). Ensure the 4 new columns are written as integers/strings, not formulas.
- Verify by re-reading the file and printing sheet names and first few rows of each sheet.

#### Step 5: Write the Word document
- Use `python-docx` to create `/root/Returns_Disposition_Brief.docx`.
- Write a short executive summary (3-6 sentences) that includes:
  - A plain-language definition of both checks: "Missing Final Event" means no completed disposition event was recorded for a return line; "Disposition Mismatch" means the final recorded disposition differs from the planned disposition.
  - The exact computed totals for Missing Final Events, Disposition Mismatches, and Total Errors (use the Grand Total row values).
  - At least one actionable recommendation (e.g., investigate root causes, improve carrier compliance, add real-time alerts).
  - Mention at least two specific Return IDs that have the most errors (highest Total Errors). To find these, group `Formatted Data` by Return ID, sum Total Errors, and pick the top 2.
- Verify the file was created.

#### Step 6: Final Validation
- Re-read `/root/Returns_Disposition_Audit.xlsx` and confirm:
  - Sheet names are exactly `['RawData', 'Formatted Data', 'Summary']`
  - `RawData` row count matches original plan table
  - `Formatted Data` has 12 columns with correct headers
  - `Formatted Data` row count matches `RawData`
  - `Summary` last row has Warehouse=`Grand Total`
  - `Summary` column headers are exactly: `Warehouse`, `Carrier`, `Missing Final Events`, `Disposition Mismatches`, `Total Errors`
- Confirm `/root/Returns_Disposition_Brief.docx` exists and is non-empty.
- Print "ALL VALIDATIONS PASSED" if everything checks out.

### Important notes:
- Install any needed packages (`openpyxl`, `python-docx`) via pip if not available.
- Do NOT use Excel formulas in the Formatted Data sheet — write static values.
- Column names and sheet names must be EXACTLY as specified (case-sensitive, spacing-sensitive).
- If any column names in the source files don't match expectations, print them and adapt accordingly.
- Print intermediate results at each step so issues can be diagnosed.

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