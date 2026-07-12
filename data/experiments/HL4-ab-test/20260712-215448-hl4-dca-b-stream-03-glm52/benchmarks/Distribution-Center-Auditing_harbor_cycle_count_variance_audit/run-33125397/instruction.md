# Task Instruction

## Task: Cycle Count Variance Audit

Create two deliverables from three input files. Follow every step carefully.

### Step 1: Inspect Input Files

```bash
cd /root
python3 -c "
import openpyxl
for fn in ['Cycle_Plan.xlsx', 'Count_Event_Log.xlsx', 'Cycle_Template.xlsx']:
    wb = openpyxl.load_workbook(fn)
    print(f'\n=== {fn} ===')
    print(f'Sheets: {wb.sheetnames}')
    for sn in wb.sheetnames:
        ws = wb[sn]
        print(f'  Sheet: {sn}, rows={ws.max_row}, cols={ws.max_column}')
        for r in range(1, min(ws.max_row+1, 8)):
            print(f'    Row {r}: {[ws.cell(r, c).value for c in range(1, ws.max_column+1)]}')
"
```

Read the output carefully. Identify:
- Which sheet in `Cycle_Plan.xlsx` has the plan table, its headers and row count.
- Which sheet in `Count_Event_Log.xlsx` has the event log, its headers (especially columns for Facility, Session ID, Bin ID, Event Type, Count Qty).
- The `Overview` sheet in `Cycle_Template.xlsx` — note its exact content, merged cells, styles.

### Step 2: Print ALL data from the event log and plan

Print ALL rows (not just first 7) from both files so you can verify your logic:

```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('Count_Event_Log.xlsx')
ws = wb.active
print('=== Count_Event_Log ===')
for r in range(1, ws.max_row+1):
    print([ws.cell(r, c).value for c in range(1, ws.max_column+1)])
"
```

```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('Cycle_Plan.xlsx')
ws = wb.active
print('=== Cycle_Plan ===')
for r in range(1, ws.max_row+1):
    print([ws.cell(r, c).value for c in range(1, ws.max_column+1)])
"
```

### Step 3: Build the output Excel file

Write a single Python script that:

1. **Reads all input data** using openpyxl.

2. **Creates `/root/Cycle_Count_Variance_Audit.xlsx`** with exactly 4 sheets in this order: `Overview`, `RawData`, `Formatted Data`, `Summary`.

3. **Overview sheet**: Copy from `Cycle_Template.xlsx` `Overview` sheet exactly — cell values, merged cells, styles (font, fill, alignment, border), column widths, row heights. Use openpyxl's copy utility. Preserve it unchanged.

4. **RawData sheet**: Copy the plan table from `Cycle_Plan.xlsx` exactly — all rows, all columns, same order, same values.

5. **Formatted Data sheet**:
   - Same rows and order as RawData.
   - First 7 columns exactly: Facility, Session ID, Bin ID, Product ID, Expected Qty, Allowed Variance, Approval Needed.
   - Process `Count_Event_Log.xlsx`:
     - Filter to rows where `Event Type` == `FINAL` (case-insensitive comparison).
     - Exclude rows where any of Facility, Session ID, Bin ID is blank/None, or Count Qty is blank/None.
     - For each unique `(Facility, Session ID, Bin ID)` key, keep only the LAST row (by row order in the spreadsheet, i.e., highest row number).
     - Build a lookup dict: key -> Count Qty.
   - For each plan row, compute:
     - `Missing Final Count`: 1 if key not in lookup dict, else 0.
     - `Approval Gap`: 1 if ALL of: (a) key IS in lookup dict, (b) `Approval Needed` is `YES` (case-insensitive, strip whitespace), (c) `abs(Expected Qty - Count Qty) > Allowed Variance`. Otherwise 0.
     - `Total Errors` = Missing Final Count + Approval Gap.
     - `Error Summary`: exactly one of `None`, `Missing Final Count`, `Approval Gap`, `Missing Final Count, Approval Gap` based on which flags are 1.
   - Write concrete values (not formulas).

6. **Summary sheet**:
   - Headers: Facility, Session ID, Missing Final Counts, Approval Gaps, Total Errors.
   - Aggregate from Formatted Data by (Facility, Session ID).
   - Include only groups where Total Errors > 0.
   - Sort by Facility ascending, then Session ID ascending.
   - Append Grand Total row: Facility=`Grand Total`, Session ID=`-`, sums of the three numeric columns across ALL included rows.

7. **Save** the workbook.

IMPORTANT implementation notes:
- When copying the Overview sheet, iterate over all cells and copy value, font, fill, border, alignment, number_format. Also copy merged_cells. Also copy column dimensions and row dimensions.
- Use `from copy import copy` for style objects.
- Make sure the sheet order is exactly: Overview, RawData, Formatted Data, Summary. If needed, reorder sheets after creation.
- Be careful with data types: Expected Qty, Allowed Variance, Count Qty should be treated as numbers. Strip strings where needed.

### Step 4: Build the Word document

Write a Python script using `python-docx` to create `/root/Cycle_Count_Variance_Brief.docx`:

- Title: "Cycle Count Variance Audit Brief" or similar.
- 3-6 sentence executive summary paragraph that includes:
  1. Plain-language definition of Missing Final Count check (bins that were scheduled for counting but never received a final count submission).
  2. Plain-language definition of Approval Gap check (bins where the final count deviated from expected quantity beyond the allowed variance threshold, requiring approval that may not have been obtained).
  3. The computed totals: X Missing Final Counts, Y Approval Gaps, Z Total Errors (use actual numbers from your computation).
  4. At least one actionable recommendation (e.g., prioritize recounting bins with missing finals, implement automated alerts).
  5. Mention at least two specific high-priority facility-session combinations that had the most exceptions (use actual facility and session ID values from Summary data).

To get the actual numbers, either compute them in the same script or read back from the Excel file.

### Step 5: Validate

After creating both files, validate:

```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('/root/Cycle_Count_Variance_Audit.xlsx')
print('Sheets:', wb.sheetnames)
assert wb.sheetnames == ['Overview', 'RawData', 'Formatted Data', 'Summary'], f'Wrong sheets: {wb.sheetnames}'

# Check Formatted Data headers
ws = wb['Formatted Data']
headers = [ws.cell(1, c).value for c in range(1, 12)]
print('Formatted Data headers:', headers)
expected = ['Facility', 'Session ID', 'Bin ID', 'Product ID', 'Expected Qty', 'Allowed Variance', 'Approval Needed', 'Missing Final Count', 'Approval Gap', 'Total Errors', 'Error Summary']
assert headers == expected, f'Header mismatch: {headers}'

# Check Summary headers
ws2 = wb['Summary']
sh = [ws2.cell(1, c).value for c in range(1, 6)]
print('Summary headers:', sh)
assert sh == ['Facility', 'Session ID', 'Missing Final Counts', 'Approval Gaps', 'Total Errors'], f'Summary header mismatch: {sh}'

# Check last row of Summary is Grand Total
last_row = ws2.max_row
print(f'Summary last row ({last_row}):', [ws2.cell(last_row, c).value for c in range(1, 6)])
assert ws2.cell(last_row, 1).value == 'Grand Total'
assert ws2.cell(last_row, 2).value == '-'

# Print some Formatted Data rows for spot check
for r in range(1, min(ws.max_row+1, 10)):
    print([ws.cell(r, c).value for c in range(1, 12)])

print('\nAll Summary rows:')
for r in range(1, ws2.max_row+1):
    print([ws2.cell(r, c).value for c in range(1, 6)])

print('\nValidation passed!')
"
```

```bash
python3 -c "
from docx import Document
doc = Document('/root/Cycle_Count_Variance_Brief.docx')
for p in doc.paragraphs:
    print(repr(p.text))
print('Word doc OK')
"
```

If any check fails, diagnose and fix before finishing.

### Step 6: Run the verifier if available

Check if there's a test script:
```bash
ls /root/test_output.py 2>/dev/null && python3 -m pytest /root/test_output.py -v
```

If it exists and any tests fail, read the failure output carefully, diagnose, and fix. Re-run until all tests pass.

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