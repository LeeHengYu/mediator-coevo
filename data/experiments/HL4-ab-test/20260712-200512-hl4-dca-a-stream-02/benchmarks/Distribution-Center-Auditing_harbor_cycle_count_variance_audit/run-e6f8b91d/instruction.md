# Task Instruction

## Task: Cycle Count Variance Audit

Create two deliverables from three input files. Follow these steps precisely.

### Step 1: Inspect Input Files

```bash
cd /root
python3 -c "
import openpyxl
for f in ['Cycle_Plan.xlsx', 'Count_Event_Log.xlsx', 'Cycle_Template.xlsx']:
    wb = openpyxl.load_workbook(f)
    print(f'=== {f} === sheets: {wb.sheetnames}')
    for sn in wb.sheetnames:
        ws = wb[sn]
        print(f'  Sheet: {sn}, rows={ws.max_row}, cols={ws.max_column}')
        for r in ws.iter_rows(min_row=1, max_row=min(5, ws.max_row), values_only=False):
            print('   ', [c.value for c in r])
    wb.close()
"
```

Also inspect the Count_Event_Log more thoroughly (all rows if feasible, or at least the headers and a sample):

```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('Count_Event_Log.xlsx')
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'Sheet: {sn}, rows={ws.max_row}, cols={ws.max_column}')
    for r in ws.iter_rows(min_row=1, max_row=ws.max_row, values_only=True):
        print(r)
wb.close()
"
```

### Step 2: Build the Output Excel and Word Files

After inspecting, write a single Python script `/root/build_audit.py` that does everything. Run it and verify.

The script must:

#### A) Load data
- Load `Cycle_Plan.xlsx` plan table into a list of dicts (note the actual column names from inspection).
- Load `Count_Event_Log.xlsx` into a list of dicts.
- Load `Cycle_Template.xlsx` to copy the `Overview` sheet.

#### B) Create `/root/Cycle_Count_Variance_Audit.xlsx` with exactly 4 sheets in order: `Overview`, `RawData`, `Formatted Data`, `Summary`.

##### Overview sheet
- Copy the `Overview` sheet from `Cycle_Template.xlsx` cell-by-cell, preserving all values and merged cells if any. Use openpyxl. Copy cell values, fonts, fills, borders, alignment, number formats if possible. At minimum, copy all cell values exactly.

##### RawData sheet
- Copy the plan table from `Cycle_Plan.xlsx` exactly (all rows, all columns, same order).

##### Formatted Data sheet
- Same rows and row order as RawData.
- First 7 columns with EXACTLY these headers (regardless of source column names): `Facility`, `Session ID`, `Bin ID`, `Product ID`, `Expected Qty`, `Allowed Variance`, `Approval Needed`.
- Map source columns to these headers based on inspection (they may have slightly different names or casing).
- Add columns 8-11: `Missing Final Count`, `Approval Gap`, `Total Errors`, `Error Summary`.

**Deriving final counts from Count_Event_Log.xlsx:**
- Filter to rows where `Event Type` (or equivalent column) equals `FINAL` (case-insensitive comparison).
- Exclude rows where any of `Facility`, `Session ID`, `Bin ID` is blank/None, or where `Count Qty` is blank/None.
- For each unique key `(Facility, Session ID, Bin ID)`, keep only the LAST row (latest by row order or timestamp if available). Store the `Count Qty` value.
- Build a lookup dict: `final_counts[(facility, session_id, bin_id)] = count_qty`.

**Computing new columns for each plan row:**
- `Missing Final Count`: 1 if key `(Facility, Session ID, Bin ID)` is NOT in `final_counts`, else 0.
- `Approval Gap`: 1 if ALL THREE conditions hold:
  1. Key IS in `final_counts` (i.e., Missing Final Count == 0)
  2. `Approval Needed` equals `YES` (case-insensitive, strip whitespace)
  3. `abs(Expected Qty - final_count_qty) > Allowed Variance` (strictly greater)
  Otherwise 0.
- `Total Errors` = `Missing Final Count + Approval Gap`
- `Error Summary`:
  - If both flags are 0: `None`
  - If only Missing Final Count is 1: `Missing Final Count`
  - If only Approval Gap is 1: `Approval Gap`
  - If both are 1: `Missing Final Count, Approval Gap`

**IMPORTANT**: Write concrete numeric/string values, NOT Excel formulas.

##### Summary sheet
- Headers: `Facility`, `Session ID`, `Missing Final Counts`, `Approval Gaps`, `Total Errors`
- Aggregate from Formatted Data by `(Facility, Session ID)`.
- Sum `Missing Final Count` → `Missing Final Counts`, sum `Approval Gap` → `Approval Gaps`, sum `Total Errors` → `Total Errors` for each group.
- Include ONLY groups where `Total Errors > 0`.
- Sort by `Facility` ascending, then `Session ID` ascending.
- Append a final row: `Grand Total`, `-`, and the sums of the three numeric columns across all included groups.

#### C) Create `/root/Cycle_Count_Variance_Brief.docx`
- Use `python-docx` library.
- Write a heading: "Cycle Count Variance Audit – Executive Summary"
- Write 3-6 sentences that include:
  1. Plain-language definition of Missing Final Count check (a bin was scheduled for counting but no final count event was recorded).
  2. Plain-language definition of Approval Gap check (a bin required approval due to variance exceeding the allowed threshold, but the discrepancy was not resolved).
  3. The computed grand totals for Missing Final Counts, Approval Gaps, and Total Errors (use actual numbers from the data).
  4. At least one actionable recommendation (e.g., "We recommend prioritizing recounts for bins with missing final counts and escalating unresolved approval gaps to warehouse supervisors.").
  5. Mention at least two specific high-priority facility-session combinations that have the most exceptions (pick the top 2 by Total Errors from the Summary data).

### Step 3: Run and Verify

Run the script:
```bash
python3 /root/build_audit.py
```

Then verify the outputs:
```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('/root/Cycle_Count_Variance_Audit.xlsx')
print('Sheets:', wb.sheetnames)
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'\n=== {sn} === rows={ws.max_row}, cols={ws.max_column}')
    for r in ws.iter_rows(min_row=1, max_row=min(8, ws.max_row), values_only=True):
        print(r)
    if ws.max_row > 8:
        print('  ...')
        for r in ws.iter_rows(min_row=ws.max_row-2, max_row=ws.max_row, values_only=True):
            print(r)
wb.close()
"
```

Verify the Word doc:
```bash
python3 -c "
from docx import Document
doc = Document('/root/Cycle_Count_Variance_Brief.docx')
for p in doc.paragraphs:
    print(repr(p.text))
"
```

### Step 4: Validate Specific Contracts

Check these critical points:
1. Sheet names are exactly `['Overview', 'RawData', 'Formatted Data', 'Summary']` in that order.
2. Formatted Data headers in row 1 are exactly: `Facility`, `Session ID`, `Bin ID`, `Product ID`, `Expected Qty`, `Allowed Variance`, `Approval Needed`, `Missing Final Count`, `Approval Gap`, `Total Errors`, `Error Summary`.
3. Summary headers are exactly: `Facility`, `Session ID`, `Missing Final Counts`, `Approval Gaps`, `Total Errors`.
4. Summary last row has `Grand Total` in first column and `-` in second.
5. All numeric columns in Formatted Data (cols 8-10) contain integers (0 or 1), not formulas.
6. Error Summary values are exactly one of: `None`, `Missing Final Count`, `Approval Gap`, `Missing Final Count, Approval Gap`.
7. Overview sheet content matches the template.
8. Word doc contains the required totals and mentions at least 2 facility-session pairs.

If any check fails, fix and re-run. Do not mark complete until all checks pass.

### Important Notes
- Install `python-docx` if not available: `pip install python-docx`
- Install `openpyxl` if not available: `pip install openpyxl`
- Pay close attention to actual column names in the source files — map them correctly to the required output headers.
- Use strict greater-than (`>`) for the variance comparison, not greater-than-or-equal.
- The `None` string in Error Summary is the literal text `None`, not a Python None value.

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