# Task Instruction

Execute the following steps in a single Python script to produce the two deliverables.

## 0. Setup
```bash
pip install openpyxl python-docx
```

## 1. Inspect source files
Before writing any logic, read and print:
- Sheet names and first 5 rows of `/root/Cycle_Plan.xlsx`
- Sheet names and first 5 rows of `/root/Count_Event_Log.xlsx`
- Sheet names and first 5 rows of `/root/Cycle_Template.xlsx`

Print column headers for each sheet so you know the exact column names (they may differ slightly from the spec names). Map them carefully.

## 2. Build `/root/Cycle_Count_Variance_Audit.xlsx`

Use `openpyxl` throughout so you can copy the Overview sheet faithfully.

### 2a. Overview sheet
- Open `Cycle_Template.xlsx` with openpyxl (data_only=False to preserve everything).
- Copy the `Overview` sheet cell-by-cell (values, merged cells, column widths, row heights if possible) into a new workbook's first sheet named exactly `Overview`.
- Preserve merged cells. Iterate over `ws.merged_cells.ranges` and replicate them.

### 2b. RawData sheet
- Read `Cycle_Plan.xlsx` into a pandas DataFrame. Identify the exact column names.
- Write this DataFrame verbatim into a sheet named exactly `RawData` (with header row).

### 2c. Formatted Data sheet
- Start from the RawData DataFrame (same row order).
- The first 7 columns must be named exactly: `Facility`, `Session ID`, `Bin ID`, `Product ID`, `Expected Qty`, `Allowed Variance`, `Approval Needed`. Rename source columns if needed to match these exact names.
- Process `Count_Event_Log.xlsx`:
  - Read into DataFrame.
  - Print its columns and dtypes.
  - Filter to rows where `Event Type` (find the exact column name) equals `FINAL` (case-insensitive compare).
  - Drop rows where any of `Facility`, `Session ID`, `Bin ID`, or `Count Qty` is NaN/blank.
  - Sort by timestamp/event date column descending (or by row order descending) so that the LAST occurrence per key is kept.
  - `drop_duplicates(subset=[facility_col, session_col, bin_col], keep='first')` to keep only the latest FINAL event per (Facility, Session ID, Bin ID).
  - Build a lookup dict: `(facility, session_id, bin_id) -> count_qty`.

- For each row in the plan DataFrame, compute:
  - `Missing Final Count`: 1 if key not in lookup dict, else 0.
  - `Approval Gap`: 1 if ALL of:
    1. Key IS in lookup dict (final count exists).
    2. `Approval Needed` value stripped and uppercased == `YES`.
    3. `abs(Expected Qty - Count Qty) > Allowed Variance` (strictly greater).
    Otherwise 0.
  - `Total Errors` = `Missing Final Count` + `Approval Gap`.
  - `Error Summary`:
    - If both flags are 1: `Missing Final Count, Approval Gap`
    - If only Missing Final Count: `Missing Final Count`
    - If only Approval Gap: `Approval Gap`
    - If neither: `None`

- Write to sheet named exactly `Formatted Data` with headers in this exact order: Facility, Session ID, Bin ID, Product ID, Expected Qty, Allowed Variance, Approval Needed, Missing Final Count, Approval Gap, Total Errors, Error Summary.
- Write concrete values (int for numeric flags, string for Error Summary), NOT formulas.

### 2d. Summary sheet
- From the Formatted Data DataFrame, group by `(Facility, Session ID)`.
- Sum `Missing Final Count`, `Approval Gap`, `Total Errors` per group.
- Filter to groups where `Total Errors > 0`.
- Sort by Facility ascending, then Session ID ascending.
- Append a Grand Total row: Facility=`Grand Total`, Session ID=`-`, and sums of the three numeric columns across ALL groups (including those with 0 errors? No — sum from the Formatted Data DataFrame overall, not just filtered groups. Actually re-read: "remaining columns = dataset totals" means totals from the entire dataset).
  - Compute grand totals from the FULL Formatted Data (all rows), not just the filtered summary rows. This ensures the Grand Total matches the full dataset.
- Column headers exactly: `Facility`, `Session ID`, `Missing Final Counts`, `Approval Gaps`, `Total Errors`.
  - Note the plural forms: `Missing Final Counts`, `Approval Gaps`.
- Write to sheet named exactly `Summary`.

### 2e. Save
Save workbook to `/root/Cycle_Count_Variance_Audit.xlsx`.

## 3. Build `/root/Cycle_Count_Variance_Brief.docx`

Using `python-docx`:
- Create a document with a heading like "Cycle Count Variance Audit – Executive Summary".
- Write 3-6 sentences that include:
  1. Plain-language definition of Missing Final Count check (a bin was scheduled for counting but no final count event was recorded).
  2. Plain-language definition of Approval Gap check (a final count was recorded but the variance between expected and actual quantity exceeded the allowed threshold for items requiring approval).
  3. The computed grand totals: X Missing Final Counts, Y Approval Gaps, Z Total Errors.
  4. Mention at least two specific (Facility, Session ID) combinations from the Summary that have the highest Total Errors (pick the top 2 by Total Errors, breaking ties by Facility then Session ID).
  5. At least one actionable recommendation (e.g., "We recommend prioritizing recounts for the flagged bins and reviewing the approval workflow for high-variance items.").
- Save to `/root/Cycle_Count_Variance_Brief.docx`.

## 4. Validation
After creating both files:
- Re-open the Excel file and print sheet names (should be exactly: Overview, RawData, Formatted Data, Summary).
- Print first 3 and last 3 rows of Formatted Data sheet.
- Print all rows of Summary sheet.
- Print the Word document paragraph texts.
- Confirm both files exist at the expected paths.

## Key Cautions
- Column name mapping: inspect actual source column names before assuming. Print them.
- The Event Log may have columns named differently (e.g., `Event Type` vs `EventType`, `Count Qty` vs `Counted Qty`). Adapt accordingly.
- Use `str.strip().upper()` for case-insensitive comparisons on Event Type and Approval Needed.
- Ensure numeric columns (Expected Qty, Allowed Variance, Count Qty) are treated as numbers, not strings. Convert if needed.
- The Overview sheet copy must preserve merged cells and content exactly.
- Do NOT leave any sheet with default names like `Sheet` or `Sheet1` in the output workbook. Delete any extra default sheets.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=expert, tags=[excel, openpyxl, docx, audit, inventory].
Verifier config: timeout_sec=900.0.