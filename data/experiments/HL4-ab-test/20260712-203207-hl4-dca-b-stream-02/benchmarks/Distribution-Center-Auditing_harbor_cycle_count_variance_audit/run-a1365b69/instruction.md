# Task Instruction

## Task: Cycle Count Variance Audit

You must create two deliverable files:
1. `/root/Cycle_Count_Variance_Audit.xlsx`
2. `/root/Cycle_Count_Variance_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect input files
```bash
pip install openpyxl python-docx pandas
```
Then use Python to inspect the three input files:
- `/root/Cycle_Plan.xlsx` — read all sheets, print sheet names, column headers, first 5 rows, shape
- `/root/Count_Event_Log.xlsx` — same inspection
- `/root/Cycle_Template.xlsx` — same inspection, pay special attention to the `Overview` sheet

Print everything so you understand the data before proceeding.

#### Step 1: Build the output Excel workbook

Use `openpyxl` to create `/root/Cycle_Count_Variance_Audit.xlsx` with exactly these sheet names in this order:
- `Overview`
- `RawData`
- `Formatted Data`
- `Summary`

##### 1a) `Overview` sheet
- Copy the `Overview` sheet from `Cycle_Template.xlsx` cell-by-cell, preserving all values, merged cells, formatting, column widths, and row heights as closely as possible. Use openpyxl to iterate over all cells and copy value, font, fill, border, alignment, number_format. Also copy merged cell ranges. The goal is an exact replica.

##### 1b) `RawData` sheet
- Copy the plan table from `Cycle_Plan.xlsx` exactly (all rows, all columns, same order). Include headers in row 1.

##### 1c) `Formatted Data` sheet
- Start with the same rows (same order) as `RawData`.
- The first 7 columns must be exactly: `Facility`, `Session ID`, `Bin ID`, `Product ID`, `Expected Qty`, `Allowed Variance`, `Approval Needed`.
- Add 4 new columns (8-11): `Missing Final Count`, `Approval Gap`, `Total Errors`, `Error Summary`.

**Deriving the final count lookup:**
1. Load `Count_Event_Log.xlsx` into a DataFrame.
2. Filter to rows where `Event Type` == `FINAL` (case-sensitive match on the column value; inspect actual values first).
3. Drop rows where any of `Facility`, `Session ID`, `Bin ID`, or `Count Qty` is blank/NaN.
4. Sort by whatever timestamp or row-order column exists (or by index) and keep only the LAST row per `(Facility, Session ID, Bin ID)` group. This gives the latest FINAL event.
5. Build a lookup dict: key = `(Facility, Session ID, Bin ID)` → value = `Count Qty`.

**Computing the 4 new columns for each plan row:**
- `Missing Final Count`: 1 if key `(Facility, Session ID, Bin ID)` is NOT in the lookup dict, else 0.
- `Approval Gap`: 1 if ALL three conditions hold:
  1. Key IS in the lookup dict (i.e., Missing Final Count == 0).
  2. `Approval Needed` value equals `YES` (case-insensitive comparison).
  3. `abs(Expected Qty - Count Qty)` is strictly greater than `Allowed Variance`.
  Otherwise 0.
- `Total Errors` = `Missing Final Count` + `Approval Gap`.
- `Error Summary`:
  - If both flags are 0: `None`
  - If Missing Final Count=1 and Approval Gap=0: `Missing Final Count`
  - If Missing Final Count=0 and Approval Gap=1: `Approval Gap`
  - If both are 1: `Missing Final Count, Approval Gap`

**CRITICAL**: Write concrete numeric values (int 0 or 1) and text strings, NOT Excel formulas.

##### 1d) `Summary` sheet
- Headers (row 1): `Facility`, `Session ID`, `Missing Final Counts`, `Approval Gaps`, `Total Errors`
- Aggregate from `Formatted Data` by `(Facility, Session ID)`:
  - `Missing Final Counts` = sum of `Missing Final Count` column for that group
  - `Approval Gaps` = sum of `Approval Gap` column for that group
  - `Total Errors` = sum of `Total Errors` column for that group
- Include ONLY groups where `Total Errors > 0`.
- Sort by `Facility` ascending, then `Session ID` ascending.
- Append a final row: `Facility`=`Grand Total`, `Session ID`=`-`, and the remaining 3 columns = grand totals across all included groups.

#### Step 2: Build the Word document

Create `/root/Cycle_Count_Variance_Brief.docx` using `python-docx`.

Content: An executive summary paragraph (3-6 sentences) that includes:
1. A plain-language definition of both checks:
   - "Missing Final Count" flags bins where no final physical count was recorded.
   - "Approval Gap" flags bins where a final count exists but the variance between expected and actual quantity exceeds the allowed threshold and requires approval.
2. The computed totals: total Missing Final Counts, total Approval Gaps, total Total Errors (use the Grand Total row values).
3. At least one actionable recommendation (e.g., prioritize recounts, tighten count procedures).
4. Mention at least two specific high-priority `(Facility, Session ID)` combinations with the most exceptions.

To identify the high-priority combinations, pick the top 2 rows from the Summary sheet by `Total Errors` descending.

#### Step 3: Verify outputs

1. Re-open `/root/Cycle_Count_Variance_Audit.xlsx` and verify:
   - Sheet names are exactly `['Overview', 'RawData', 'Formatted Data', 'Summary']`
   - `RawData` row count matches `Cycle_Plan.xlsx`
   - `Formatted Data` has 11 columns with correct headers
   - `Summary` has 5 columns with correct headers, only error groups, sorted correctly, Grand Total row at end
   - Print a few sample rows from each sheet
2. Re-open `/root/Cycle_Count_Variance_Brief.docx` and print its text to confirm content.
3. Check that both files exist at the correct paths.

#### Step 4: Run the verifier if available
```bash
ls /root/test_output.py 2>/dev/null && cd /root && python -m pytest test_output.py -v
```
If the verifier exists, run it and fix any failures. If specific assertions fail, read the test code to understand what's expected and adjust accordingly.

### Important Notes
- Do NOT delete or rename the default sheet that openpyxl creates; instead, create sheets in order and remove the default sheet at the end if it's not one of the 4 required sheets.
- Be very careful with sheet name casing and spacing: `Formatted Data` has a space, `RawData` does not.
- When copying the Overview sheet, if there are merged cells, copy them. If copying fails on styling, at minimum preserve all cell values.
- Ensure all numeric columns (`Missing Final Count`, `Approval Gap`, `Total Errors`, `Missing Final Counts`, `Approval Gaps`) contain Python `int` values, not floats or strings.
- `Error Summary` values must be exact strings: `None`, `Missing Final Count`, `Approval Gap`, or `Missing Final Count, Approval Gap` (note the comma and space).

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