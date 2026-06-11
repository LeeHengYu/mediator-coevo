# Task Instruction

## Task: Cycle Count Variance Audit

Create two deliverable files from three input files. Follow these steps precisely.

### Step 1: Inspect Input Files

```bash
cd /root
python3 -c "
import openpyxl

# Inspect Cycle_Plan.xlsx
wb = openpyxl.load_workbook('Cycle_Plan.xlsx')
for s in wb.sheetnames:
    ws = wb[s]
    print(f'=== Cycle_Plan / {s} ===')
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        print(row)
        if i > 5: break
wb.close()

# Inspect Count_Event_Log.xlsx
wb = openpyxl.load_workbook('Count_Event_Log.xlsx')
for s in wb.sheetnames:
    ws = wb[s]
    print(f'=== Count_Event_Log / {s} ===')
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        print(row)
        if i > 10: break
wb.close()

# Inspect Cycle_Template.xlsx
wb = openpyxl.load_workbook('Cycle_Template.xlsx')
for s in wb.sheetnames:
    ws = wb[s]
    print(f'=== Cycle_Template / {s} ===')
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        print(row)
        if i > 10: break
wb.close()
"
```

Carefully note:
- The exact column headers and order in Cycle_Plan.xlsx
- The exact column headers in Count_Event_Log.xlsx (especially the column names for Facility, Session ID, Bin ID, Event Type, Count Qty, and any timestamp/sequence column)
- The Overview sheet content in Cycle_Template.xlsx

### Step 2: Build the Solution Script

After inspecting, write a single Python script `/root/build_audit.py` that does everything below. Use `openpyxl` for Excel and `python-docx` for Word. Install if needed: `pip install openpyxl python-docx`.

#### 2a: Load Data with pandas
- Read `Cycle_Plan.xlsx` into a DataFrame `plan_df`. Keep all columns.
- Read `Count_Event_Log.xlsx` into a DataFrame `events_df`.

#### 2b: Derive Final Counts
- From `events_df`, filter to rows where `Event Type` equals `FINAL` (case-insensitive match; strip whitespace).
- Drop rows where any of Facility, Session ID, or Bin ID is blank/NaN.
- Drop rows where `Count Qty` is blank/NaN.
- If there's a timestamp or sequence column, sort by it descending and keep the first (latest) row per `(Facility, Session ID, Bin ID)`. If no such column, keep the last row per group.
- Create a lookup dictionary: key = `(Facility, Session ID, Bin ID)` → value = `Count Qty`.

#### 2c: Build Formatted Data
- Start with `plan_df` rows in original order.
- The first 7 columns must be exactly: `Facility`, `Session ID`, `Bin ID`, `Product ID`, `Expected Qty`, `Allowed Variance`, `Approval Needed`. Use the actual column names from the file but rename if needed to match these exact headers.
- For each row, compute:
  - `final_count_qty` = lookup value for `(Facility, Session ID, Bin ID)`, or None if missing.
  - `Missing Final Count` = 1 if `final_count_qty` is None, else 0.
  - `Approval Gap` = 1 if ALL of: (a) `final_count_qty` is not None, (b) `Approval Needed` equals `YES` (case-insensitive, stripped), (c) `abs(Expected Qty - final_count_qty) > Allowed Variance`. Otherwise 0.
  - `Total Errors` = `Missing Final Count + Approval Gap`.
  - `Error Summary`: build from the flags. If both 0 → `None`. If only missing → `Missing Final Count`. If only approval → `Approval Gap`. If both → `Missing Final Count, Approval Gap`.
- Store as list of lists for writing.

#### 2d: Build Summary
- Group `Formatted Data` by `(Facility, Session ID)`.
- For each group, sum `Missing Final Count`, `Approval Gap`, `Total Errors`.
- Keep only groups with `Total Errors > 0`.
- Sort by Facility ascending, then Session ID ascending.
- Append a Grand Total row: Facility=`Grand Total`, Session ID=`-`, sums of the three numeric columns.
- Headers: `Facility`, `Session ID`, `Missing Final Counts`, `Approval Gaps`, `Total Errors`.

#### 2e: Write Excel Output
Create `/root/Cycle_Count_Variance_Audit.xlsx` with openpyxl:

1. **Overview sheet**: Copy from `Cycle_Template.xlsx` cell by cell (values AND merged cells if any). Preserve formatting where possible using `copy` from openpyxl styles. The sheet must be named exactly `Overview`.
2. **RawData sheet**: Write the plan table exactly as read from `Cycle_Plan.xlsx` (headers + data). Named exactly `RawData`.
3. **Formatted Data sheet**: Write headers and computed data. Named exactly `Formatted Data`.
4. **Summary sheet**: Write headers and summary data. Named exactly `Summary`.

**Critical**: Ensure sheet order is `Overview`, `RawData`, `Formatted Data`, `Summary`. Remove any default sheets.

**Data type care**:
- Write numeric values as Python `int` or `float`, not strings.
- Write `Missing Final Count`, `Approval Gap`, `Total Errors` as integers (0 or 1, or sums).
- Write `Error Summary` as a string.
- If any date columns exist, convert them to `YYYY-MM-DD` string format before writing (lesson from cross-task feedback).
- Ensure `Facility`, `Session ID`, `Bin ID`, `Product ID` are written as strings.
- Ensure `Expected Qty`, `Allowed Variance` are written as numbers.
- Ensure `Approval Needed` is written as a string.

#### 2f: Write Word Document
Create `/root/Cycle_Count_Variance_Brief.docx` with python-docx:
- Title: "Cycle Count Variance Audit – Executive Brief" (or similar)
- 3-6 sentence executive summary paragraph that includes:
  1. Plain-language definition of Missing Final Count check (a bin was scheduled for counting but no final count event was recorded).
  2. Plain-language definition of Approval Gap check (a final count was recorded but the variance between expected and actual quantity exceeded the allowed threshold for items requiring approval).
  3. The computed totals: total Missing Final Counts, total Approval Gaps, total Total Errors (use the Grand Total row values).
  4. At least one actionable recommendation (e.g., "We recommend prioritizing recounts for bins with missing final counts and escalating approval-gap items to supervisors").
  5. Mention at least two specific facility-session combinations from the Summary that have the highest Total Errors.

### Step 3: Run the Script
```bash
cd /root && python3 build_audit.py
```

### Step 4: Validate Output
```bash
python3 -c "
import openpyxl

wb = openpyxl.load_workbook('Cycle_Count_Variance_Audit.xlsx')
print('Sheet names:', wb.sheetnames)
assert wb.sheetnames == ['Overview', 'RawData', 'Formatted Data', 'Summary'], f'Wrong sheets: {wb.sheetnames}'

# Check Formatted Data headers
ws = wb['Formatted Data']
headers = [c.value for c in ws[1]]
print('Formatted Data headers:', headers)
expected_h = ['Facility', 'Session ID', 'Bin ID', 'Product ID', 'Expected Qty', 'Allowed Variance', 'Approval Needed', 'Missing Final Count', 'Approval Gap', 'Total Errors', 'Error Summary']
assert headers == expected_h, f'Header mismatch: {headers}'

# Check a few data cells are correct types
for row in ws.iter_rows(min_row=2, max_row=3, values_only=True):
    print(row)
    assert isinstance(row[7], int), f'Missing Final Count not int: {type(row[7])} = {row[7]}'
    assert isinstance(row[8], int), f'Approval Gap not int: {type(row[8])} = {row[8]}'
    assert isinstance(row[9], int), f'Total Errors not int: {type(row[9])} = {row[9]}'
    assert isinstance(row[10], str), f'Error Summary not str: {type(row[10])} = {row[10]}'

# Check Summary headers
ws2 = wb['Summary']
sh = [c.value for c in ws2[1]]
print('Summary headers:', sh)
assert sh == ['Facility', 'Session ID', 'Missing Final Counts', 'Approval Gaps', 'Total Errors'], f'Summary header mismatch: {sh}'

# Check last row is Grand Total
last_row = list(ws2.iter_rows(values_only=True))[-1]
print('Last summary row:', last_row)
assert last_row[0] == 'Grand Total', f'Last row not Grand Total: {last_row[0]}'
assert last_row[1] == '-', f'Last row Session ID not -: {last_row[1]}'

print('All validations passed!')
wb.close()
"
```

Also verify the Word doc exists:
```bash
python3 -c "
from docx import Document
doc = Document('/root/Cycle_Count_Variance_Brief.docx')
for p in doc.paragraphs:
    print(p.text)
print('Word doc OK')
"
```

### Step 5: Debug and Fix
If any validation fails, read the error, inspect the relevant data, fix `build_audit.py`, and re-run. Repeat until all validations pass.

### Key Pitfalls to Avoid
- Do NOT leave datetime objects in cells; convert to strings.
- Do NOT use Excel formulas for the computed columns; write concrete values.
- Do NOT change the Overview sheet content from the template.
- Ensure `Error Summary` uses exactly the strings: `None`, `Missing Final Count`, `Approval Gap`, or `Missing Final Count, Approval Gap` (note the comma-space separator).
- The `Error Summary` string `None` must be the literal string `"None"`, not Python's `None`/null.
- Summary column headers use plural forms: `Missing Final Counts`, `Approval Gaps` (with 's').
- Sort Summary by Facility asc, then Session ID asc. Session ID sorting should be consistent (if numeric, sort numerically; if string, sort lexicographically).
- When copying the Overview sheet, preserve merged cells if present.

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