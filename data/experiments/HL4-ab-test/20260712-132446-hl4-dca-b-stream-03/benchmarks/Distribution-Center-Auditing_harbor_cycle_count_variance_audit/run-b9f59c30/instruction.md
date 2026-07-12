# Task Instruction

## Task: Cycle Count Variance Audit

You must create two output files:
1. `/root/Cycle_Count_Variance_Audit.xlsx`
2. `/root/Cycle_Count_Variance_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect Input Files
1. Read `/root/Cycle_Plan.xlsx` — note all sheet names, column headers, row count, and data types.
2. Read `/root/Count_Event_Log.xlsx` — note all sheet names, column headers, row count. Pay special attention to columns: `Facility`, `Session ID`, `Bin ID`, `Event Type`, `Count Qty`, and any timestamp/sequence column.
3. Read `/root/Cycle_Template.xlsx` — note all sheet names. The `Overview` sheet must be copied verbatim.
4. Print the first 5 rows and all column names of each file for verification.

#### Step 1: Build the `RawData` sheet
- Load the plan table from `Cycle_Plan.xlsx` into a DataFrame. This will become the `RawData` sheet, preserving exact row order and all columns.

#### Step 2: Process `Count_Event_Log.xlsx` for final counts
- Filter rows where `Event Type` equals `FINAL` (case-insensitive comparison recommended).
- Drop rows where any of `Facility`, `Session ID`, `Bin ID`, or `Count Qty` is blank/NaN.
- For each unique `(Facility, Session ID, Bin ID)` group, keep only the **last** row (by original row order or timestamp if available). This gives the latest FINAL event per key.
- Store the result as a lookup dictionary: key = `(Facility, Session ID, Bin ID)` → value = `Count Qty`.

#### Step 3: Build the `Formatted Data` sheet
- Start from the `RawData` DataFrame (same row order).
- Ensure the first 7 columns are exactly: `Facility`, `Session ID`, `Bin ID`, `Product ID`, `Expected Qty`, `Allowed Variance`, `Approval Needed`. If the source columns have different names, rename them to match exactly.
- For each row, compute:
  - **Missing Final Count**: 1 if no FINAL event exists in the lookup for that row's `(Facility, Session ID, Bin ID)`, else 0.
  - **Approval Gap**: 1 if ALL three conditions hold:
    1. A FINAL event exists (i.e., Missing Final Count == 0)
    2. `Approval Needed` equals `YES` (case-insensitive)
    3. `abs(Expected Qty - Count Qty)` is **strictly greater than** `Allowed Variance`
    Otherwise 0.
  - **Total Errors**: `Missing Final Count + Approval Gap`
  - **Error Summary**: Exactly one of:
    - `None` (if Total Errors == 0)
    - `Missing Final Count` (if only that flag is 1)
    - `Approval Gap` (if only that flag is 1)
    - `Missing Final Count, Approval Gap` (if both are 1)
- Write concrete numeric values (int) for Missing Final Count, Approval Gap, Total Errors. Write concrete strings for Error Summary. Do NOT use Excel formulas.

#### Step 4: Build the `Summary` sheet
- Aggregate from `Formatted Data` by `(Facility, Session ID)`:
  - `Missing Final Counts` = sum of `Missing Final Count` in group
  - `Approval Gaps` = sum of `Approval Gap` in group
  - `Total Errors` = sum of `Total Errors` in group
- **Include only groups where `Total Errors > 0`**.
- Sort by `Facility` ascending, then `Session ID` ascending.
- Append a Grand Total row: `Facility` = `Grand Total`, `Session ID` = `-`, numeric columns = dataset-wide totals (sum across ALL rows in Formatted Data, not just filtered groups — but since filtered groups are exactly those with errors > 0, and groups with 0 errors contribute 0, the totals are the same).
- Headers must be exactly: `Facility`, `Session ID`, `Missing Final Counts`, `Approval Gaps`, `Total Errors`.

#### Step 5: Write `/root/Cycle_Count_Variance_Audit.xlsx`
Use `openpyxl` to create the workbook with exactly these sheets in this order:
1. `Overview` — copied from `Cycle_Template.xlsx` cell-by-cell (values, merged cells, formatting if feasible; at minimum copy all cell values exactly).
2. `RawData` — the plan table.
3. `Formatted Data` — the enriched table with 11 columns.
4. `Summary` — the aggregated table.

**Important**: Remove any default sheets (like `Sheet`) that openpyxl creates. The workbook must contain exactly these 4 sheet names.

To copy the Overview sheet faithfully, use openpyxl to load `Cycle_Template.xlsx` and iterate over all cells in the Overview sheet, copying values (and ideally styles/merged cells) to the new workbook's Overview sheet.

#### Step 6: Write `/root/Cycle_Count_Variance_Brief.docx`
Use `python-docx` to create a Word document with an executive summary paragraph (3-6 sentences) that includes:
- A plain-language definition of `Missing Final Count` (a planned bin had no final count event recorded) and `Approval Gap` (a counted bin required approval because its variance exceeded the allowed threshold, but the discrepancy was not flagged/resolved).
- The computed totals: total Missing Final Counts, total Approval Gaps, and total Total Errors (use the Grand Total row values).
- At least one actionable recommendation (e.g., "Implement mandatory final-count verification before session close").
- Mention at least two specific high-priority `(Facility, Session ID)` combinations that had the most exceptions (pick the top 2 by Total Errors from the Summary sheet).

#### Step 7: Validate
1. Re-open `/root/Cycle_Count_Variance_Audit.xlsx` with openpyxl and verify:
   - Exactly 4 sheets with exact names: `Overview`, `RawData`, `Formatted Data`, `Summary`.
   - `Formatted Data` has 11 columns with correct headers.
   - `Summary` has 5 columns with correct headers.
   - The last row of `Summary` has `Facility` = `Grand Total` and `Session ID` = `-`.
   - Print the Summary sheet contents for visual verification.
2. Re-open `/root/Cycle_Count_Variance_Brief.docx` and print its text to verify content.
3. Check that no extra sheets exist.

### Technical Notes
- Install any needed packages: `pip install openpyxl python-docx pandas`
- When copying the Overview sheet, if `Cycle_Template.xlsx` has merged cells, replicate them with `ws.merge_cells()`.
- Be careful with data type matching when creating lookup keys — convert Facility, Session ID, Bin ID to strings (stripped) for consistent matching between the plan and event log.
- If `Count Qty` in the event log is stored as a string, convert to numeric before computing variance.

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