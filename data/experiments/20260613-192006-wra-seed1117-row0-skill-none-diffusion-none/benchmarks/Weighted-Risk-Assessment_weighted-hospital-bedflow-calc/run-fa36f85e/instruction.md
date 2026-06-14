# Task Instruction

Execute the following Python script in a single step. It inspects the workbook, then writes all required formulas, and saves the result.

```python
import openpyxl
import os
import shutil

# --- Phase 0: Copy workbook and inspect Data sheet layout ---
os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')

wb = openpyxl.load_workbook('/root/output/result.xlsx')
task = wb['Task']
data = wb['Data']

# Print Task sheet structure for debugging
print('=== Task sheet header row 10 (columns H-L) ===')
for c in range(8, 13):  # H=8 .. L=12
    print(f'  col {c} ({openpyxl.utils.get_column_letter(c)}): {task.cell(row=10, column=c).value}')

print('\n=== Task sheet column D (series codes) ===')
for r in range(12, 32):
    print(f'  row {r}: D={task.cell(row=r, column=4).value}')

print('\n=== Task sheet rows 35-40 column B-D ===')
for r in range(35, 41):
    print(f'  row {r}: B={task.cell(row=r, column=2).value}, C={task.cell(row=r, column=3).value}, D={task.cell(row=r, column=4).value}')

print('\n=== Task sheet rows 42-47 column B-G ===')
for r in range(42, 48):
    print(f'  row {r}: B={task.cell(row=r, column=2).value}, C={task.cell(row=r, column=3).value}, G={task.cell(row=r, column=7).value}')

print('\n=== Task sheet row 50 ===')
for c in range(1, 13):
    print(f'  col {c}: {task.cell(row=50, column=c).value}')

print('\n=== Data sheet rows 19-40, first 20 cols ===')
for r in range(19, 41):
    vals = []
    for c in range(1, 21):
        vals.append(str(data.cell(row=r, column=c).value))
    print(f'  row {r}: {", ".join(vals)}')

print('\n=== Data sheet row 20 (likely header) ===')
for c in range(1, 21):
    print(f'  col {c} ({openpyxl.utils.get_column_letter(c)}): {data.cell(row=20, column=c).value}')

wb.close()
```

After inspecting the output, execute the following script that writes all formulas. Adjust the Data sheet range references if the inspection reveals different column/row layouts than assumed below. The script assumes:
- Data sheet rows 21:38 contain the lookup data
- Data sheet column A (or the first column) contains the series codes
- Data sheet row 20 contains year headers
- The year values in Task row 10 (H10:L10) match Data sheet column headers

```python
import openpyxl
import os

wb = openpyxl.load_workbook('/root/output/result.xlsx')
task = wb['Task']
data = wb['Data']

# --- Determine Data sheet layout from inspection ---
# Find which column in Data sheet has the series codes and which row has years
# We need to identify:
#   - The column in Data that holds series codes (to use in MATCH for rows)
#   - The row in Data that holds year values (to use in MATCH for columns)
#   - The data range for INDEX

# Detect: scan Data row 20 for year-like values to find the header row
# and scan Data column A for series codes
data_code_col = None
data_year_row = None
data_first_data_col = None
data_last_data_col = None

# Check row 20 for years
for c in range(1, 30):
    v = data.cell(row=20, column=c).value
    if v is not None and isinstance(v, (int, float)) and 1900 < v < 2100:
        if data_year_row is None:
            data_year_row = 20
            data_first_data_col = c
        data_last_data_col = c

# If not row 20, try row 19
if data_year_row is None:
    for c in range(1, 30):
        v = data.cell(row=19, column=c).value
        if v is not None and isinstance(v, (int, float)) and 1900 < v < 2100:
            if data_year_row is None:
                data_year_row = 19
                data_first_data_col = c
            data_last_data_col = c

# Find code column: check which column in rows 21-38 has text matching Task D12
test_code = task.cell(row=12, column=4).value
for c in range(1, 10):
    for r in range(21, 39):
        if data.cell(row=r, column=c).value == test_code:
            data_code_col = c
            break
    if data_code_col is not None:
        break

print(f'Data year_row={data_year_row}, code_col={data_code_col}, first_data_col={data_first_data_col}, last_data_col={data_last_data_col}')

# Build column letter references
from openpyxl.utils import get_column_letter

code_col_letter = get_column_letter(data_code_col)
first_dcol_letter = get_column_letter(data_first_data_col)
last_dcol_letter = get_column_letter(data_last_data_col)

# Data range for INDEX: rows 21:38, columns first_data_col:last_data_col
# Code range for MATCH (row lookup): Data!code_col21:code_col38
# Year range for MATCH (col lookup): Data!first_dcol_year_row:last_dcol_year_row

data_array = f"Data!${first_dcol_letter}$21:${last_dcol_letter}$38"
code_range = f"Data!${code_col_letter}$21:${code_col_letter}$38"
year_range = f"Data!${first_dcol_letter}${data_year_row}:${last_dcol_letter}${data_year_row}"

print(f'data_array={data_array}')
print(f'code_range={code_range}')
print(f'year_range={year_range}')

# --- Step 1: Populate H12:L17, H19:L24, H26:L31 with INDEX/MATCH formulas ---
blocks = [
    list(range(12, 18)),  # rows 12-17
    list(range(19, 25)),  # rows 19-24
    list(range(26, 32)),  # rows 26-31
]

for block_rows in blocks:
    for row in block_rows:
        series_code_ref = f"$D{row}"  # series code in column D
        for col_idx in range(8, 13):  # H=8 to L=12
            year_ref = f"{get_column_letter(col_idx)}$10"  # year in row 10
            formula = f'=INDEX({data_array},MATCH({series_code_ref},{code_range},0),MATCH({year_ref},{year_range},0))'
            task.cell(row=row, column=col_idx).value = formula

print('Step 1 formulas written.')

# --- Step 2: Net patient flow in H35:L40 ---
# Formula: (Patient Admissions - Patient Discharges) / Effective Bed Capacity * 100
# Patient Admissions = rows 12:17 (block 1)
# Patient Discharges = rows 19:24 (block 2)
# Effective Bed Capacity = rows 26:31 (block 3)

for i in range(6):  # 6 hospitals
    adm_row = 12 + i
    dis_row = 19 + i
    cap_row = 26 + i
    net_row = 35 + i
    for col_idx in range(8, 13):
        col_letter = get_column_letter(col_idx)
        formula = f'=({col_letter}{adm_row}-{col_letter}{dis_row})/{col_letter}{cap_row}*100'
        task.cell(row=net_row, column=col_idx).value = formula

print('Step 2 net flow formulas written.')

# --- Step 2b: Summary statistics in H42:L47 ---
# Row 42: MIN, Row 43: MAX, Row 44: MEDIAN, Row 45: AVERAGE, Row 46: 25th percentile, Row 47: 75th percentile
# But let's verify the labels first
for r in range(42, 48):
    print(f'  row {r} label: {task.cell(row=r, column=7).value} / {task.cell(row=r, column=2).value}')

# We'll assign based on typical ordering: min, max, median, mean, 25th, 75th
# But we should check labels. For now, use the standard order and adjust if needed.
stat_formulas = [
    'MIN',       # row 42
    'MAX',       # row 43
    'MEDIAN',    # row 44
    'AVERAGE',   # row 45
    'PERCENTILE',  # row 46 - 25th
    'PERCENTILE',  # row 47 - 75th
]

for idx, stat_row in enumerate(range(42, 48)):
    for col_idx in range(8, 13):
        col_letter = get_column_letter(col_idx)
        data_range = f'{col_letter}35:{col_letter}40'
        if idx < 4:  # MIN, MAX, MEDIAN, AVERAGE
            formula = f'={stat_formulas[idx]}({data_range})'
        elif idx == 4:  # 25th percentile
            formula = f'=PERCENTILE({data_range},0.25)'
        else:  # 75th percentile
            formula = f'=PERCENTILE({data_range},0.75)'
        task.cell(row=stat_row, column=col_idx).value = formula

print('Step 2b summary statistics written.')

# --- Step 3: Weighted mean in H50:L50 ---
# SUMPRODUCT(net_flow * capacity) / SUM(capacity)
# net_flow = H35:H40 .. L35:L40
# capacity = H26:H31 .. L26:L31

for col_idx in range(8, 13):
    col_letter = get_column_letter(col_idx)
    net_range = f'{col_letter}35:{col_letter}40'
    cap_range = f'{col_letter}26:{col_letter}31'
    formula = f'=SUMPRODUCT({net_range},{cap_range})/SUM({cap_range})'
    task.cell(row=50, column=col_idx).value = formula

print('Step 3 weighted mean written.')

# --- Save ---
wb.save('/root/output/result.xlsx')
wb.close()
print('Workbook saved to /root/output/result.xlsx')
```

After the second script runs, verify the formulas were written by running:

```python
import openpyxl
wb = openpyxl.load_workbook('/root/output/result.xlsx')
task = wb['Task']
print('H12:', task['H12'].value)
print('H35:', task['H35'].value)
print('H42:', task['H42'].value)
print('H50:', task['H50'].value)
print('L17:', task['L17'].value)
print('L40:', task['L40'].value)
print('L47:', task['L47'].value)
print('L50:', task['L50'].value)
wb.close()
```

IMPORTANT NOTES:
1. After Phase 0 inspection, if the Data sheet layout differs from assumptions (e.g., code column is not column A, year header row is not 20, data columns differ), adjust the second script accordingly before running it.
2. Use plain `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid #NAME? errors.
3. The summary statistics row order (min/max/median/mean/25th/75th) should match the labels visible in column G or B of rows 42-47. Check the Phase 0 output and reorder if needed.
4. Make sure every `task.cell(...).value = formula` assignment actually executes - do not skip any cells.
5. The final verification step must show formula strings (starting with '=') in all checked cells, not 'None'.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=easy, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.