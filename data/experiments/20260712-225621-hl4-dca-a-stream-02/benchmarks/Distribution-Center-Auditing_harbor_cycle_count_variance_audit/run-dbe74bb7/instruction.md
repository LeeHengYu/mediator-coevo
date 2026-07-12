# Task Instruction

## Task: Cycle Count Variance Audit

You must create two output files:
1. `/root/Cycle_Count_Variance_Audit.xlsx`
2. `/root/Cycle_Count_Variance_Brief.docx`

### Step 0: Inspect Input Files

Before writing any code, inspect all three input files to understand their structure:
```python
import openpyxl
import pandas as pd

# Inspect Cycle_Plan.xlsx
wb = openpyxl.load_workbook('/root/Cycle_Plan.xlsx')
print('Cycle_Plan sheets:', wb.sheetnames)
for s in wb.sheetnames:
    ws = wb[s]
    print(f'  Sheet {s}: {ws.max_row} rows x {ws.max_column} cols')
    for row in ws.iter_rows(min_row=1, max_row=min(5, ws.max_row), values_only=False):
        print('   ', [c.value for c in row])
wb.close()

# Inspect Count_Event_Log.xlsx
wb = openpyxl.load_workbook('/root/Count_Event_Log.xlsx')
print('Count_Event_Log sheets:', wb.sheetnames)
for s in wb.sheetnames:
    ws = wb[s]
    print(f'  Sheet {s}: {ws.max_row} rows x {ws.max_column} cols')
    for row in ws.iter_rows(min_row=1, max_row=min(5, ws.max_row), values_only=False):
        print('   ', [c.value for c in row])
wb.close()

# Inspect Cycle_Template.xlsx
wb = openpyxl.load_workbook('/root/Cycle_Template.xlsx')
print('Cycle_Template sheets:', wb.sheetnames)
for s in wb.sheetnames:
    ws = wb[s]
    print(f'  Sheet {s}: {ws.max_row} rows x {ws.max_column} cols')
    for row in ws.iter_rows(min_row=1, max_row=min(5, ws.max_row), values_only=False):
        print('   ', [c.value for c in row])
wb.close()
```

Also read them with pandas to see column names and dtypes:
```python
df_plan = pd.read_excel('/root/Cycle_Plan.xlsx')
print('Cycle_Plan columns:', list(df_plan.columns))
print(df_plan.dtypes)
print(df_plan.head())

df_log = pd.read_excel('/root/Count_Event_Log.xlsx')
print('Count_Event_Log columns:', list(df_log.columns))
print(df_log.dtypes)
print(df_log.head())
```

### Step 1: Write the Python Script

After inspecting, write a single Python script `/root/solution.py` that does all the work. The script must:

#### 1a) Copy Overview sheet from template
Use openpyxl to copy the `Overview` sheet from `Cycle_Template.xlsx` **exactly** (cell values, merges, styles if possible) into the output workbook as the first sheet named `Overview`.

#### 1b) Create RawData sheet
Copy the plan table from `Cycle_Plan.xlsx` exactly into a sheet named `RawData`. Preserve all values. **Important**: If any columns contain dates or datetimes, convert them to strings in `YYYY-MM-DD` format before writing to avoid datetime object issues.

#### 1c) Create Formatted Data sheet
Start with the same data and row order as RawData. The first 7 columns must be exactly:
1. Facility
2. Session ID
3. Bin ID
4. Product ID
5. Expected Qty
6. Allowed Variance
7. Approval Needed

**Map column names from the source file to these exact names** if they differ (e.g., strip whitespace, handle case differences). Use the inspection output to determine the exact mapping.

Then derive the 4 new columns:

**Processing Count_Event_Log.xlsx:**
- Read the event log
- Filter to rows where `Event Type` (or equivalent column) equals `FINAL` (case-insensitive comparison)
- Drop rows where any of the key columns (`Facility`, `Session ID`, `Bin ID`) are blank/NaN, or where `Count Qty` is blank/NaN
- For each unique `(Facility, Session ID, Bin ID)` group, keep only the **last** row (latest). If there's a timestamp column, sort by it first; otherwise use the natural row order
- This gives you a lookup: key → final Count Qty

**Computing new columns for each row in Formatted Data:**
- `Missing Final Count`: 1 if no kept FINAL event exists for that row's `(Facility, Session ID, Bin ID)`, else 0
- `Approval Gap`: 1 if ALL three conditions hold:
  1. A kept final event exists (Missing Final Count == 0)
  2. `Approval Needed` == `YES` (case-insensitive, strip whitespace)
  3. `abs(Expected Qty - Count Qty)` is **strictly greater than** `Allowed Variance`
  Otherwise 0.
- `Total Errors` = `Missing Final Count` + `Approval Gap`
- `Error Summary`: exactly one of:
  - `None` (if Total Errors == 0)
  - `Missing Final Count` (if only that flag is 1)
  - `Approval Gap` (if only that flag is 1)
  - `Missing Final Count, Approval Gap` (if both are 1)

**Critical**: Write concrete numeric values (int 0 or 1) and text strings, NOT Excel formulas.

**Critical**: Convert any date/datetime columns to `YYYY-MM-DD` strings before writing.

#### 1d) Create Summary sheet
Aggregate from Formatted Data by `(Facility, Session ID)`:
- `Missing Final Counts` = sum of `Missing Final Count` for that group
- `Approval Gaps` = sum of `Approval Gap` for that group  
- `Total Errors` = sum of `Total Errors` for that group

Include **only** groups where `Total Errors > 0`.

Sort by `Facility` ascending, then `Session ID` ascending.

Append a Grand Total row:
- `Facility` = `Grand Total`
- `Session ID` = `-`
- Remaining columns = sums across all included rows

Headers must be exactly: `Facility`, `Session ID`, `Missing Final Counts`, `Approval Gaps`, `Total Errors`

#### 1e) Write the Excel file
Use openpyxl to write all four sheets in order: `Overview`, `RawData`, `Formatted Data`, `Summary`.

Make sure the sheet order and names are exactly as specified. The output file must be `/root/Cycle_Count_Variance_Audit.xlsx`.

#### 1f) Create Word document
Use python-docx to create `/root/Cycle_Count_Variance_Brief.docx` with an executive summary (3-6 sentences) that includes:
- Plain-language definition of both checks: Missing Final Count means a bin had no final count event recorded; Approval Gap means a bin's count deviated beyond the allowed variance and required approval
- The computed totals for Missing Final Counts, Approval Gaps, and Total Errors (use the actual numbers from the Grand Total row)
- At least one actionable recommendation
- Mention at least two specific high-priority facility-session combinations (pick the ones with the highest Total Errors from the Summary)

### Step 2: Run and Validate

Run the script:
```bash
python /root/solution.py
```

Then validate the outputs:
```python
import openpyxl
wb = openpyxl.load_workbook('/root/Cycle_Count_Variance_Audit.xlsx')
print('Output sheets:', wb.sheetnames)
assert wb.sheetnames == ['Overview', 'RawData', 'Formatted Data', 'Summary'], f'Sheet names mismatch: {wb.sheetnames}'

# Check Formatted Data headers
ws = wb['Formatted Data']
headers = [ws.cell(1, c).value for c in range(1, ws.max_column+1)]
print('Formatted Data headers:', headers)
expected_headers = ['Facility', 'Session ID', 'Bin ID', 'Product ID', 'Expected Qty', 'Allowed Variance', 'Approval Needed', 'Missing Final Count', 'Approval Gap', 'Total Errors', 'Error Summary']
assert headers == expected_headers, f'Header mismatch: {headers}'

# Check Summary headers
ws2 = wb['Summary']
sum_headers = [ws2.cell(1, c).value for c in range(1, ws2.max_column+1)]
print('Summary headers:', sum_headers)
assert sum_headers == ['Facility', 'Session ID', 'Missing Final Counts', 'Approval Gaps', 'Total Errors']

# Check Grand Total row
last_row = ws2.max_row
print('Grand Total row:', [ws2.cell(last_row, c).value for c in range(1, 6)])
assert ws2.cell(last_row, 1).value == 'Grand Total'
assert ws2.cell(last_row, 2).value == '-'

# Check no datetime objects in cells
for row in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True):
    for val in row:
        if hasattr(val, 'strftime'):
            print(f'WARNING: datetime object found: {val}')

# Spot check some Formatted Data values
for r in range(2, min(6, ws.max_row+1)):
    row_vals = [ws.cell(r, c).value for c in range(1, ws.max_column+1)]
    print(f'Row {r}:', row_vals)
    # Verify Total Errors = Missing Final Count + Approval Gap
    assert row_vals[9] == row_vals[7] + row_vals[8], f'Total Errors mismatch in row {r}'

wb.close()

# Check Word doc exists and has content
from docx import Document
doc = Document('/root/Cycle_Count_Variance_Brief.docx')
text = ' '.join([p.text for p in doc.paragraphs])
print('Word doc text:', text[:500])
assert 'Missing Final Count' in text
assert 'Approval Gap' in text
print('All validations passed!')
```

### Step 3: Fix any issues
If any validation fails, inspect the specific failure, fix the script, and re-run. Pay special attention to:
- Column name mapping from source files (they may not exactly match the required output names)
- Date/datetime conversion to strings
- Case-insensitive comparisons for `Event Type` and `Approval Needed`
- The distinction between `Missing Final Counts` (Summary header, plural) vs `Missing Final Count` (Formatted Data header, singular)
- Grand Total row computation should sum ALL included summary rows' values

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