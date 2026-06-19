# Task Instruction

Execute the following Python script to inspect the workbook, build the required formulas, and save the result.

```python
import openpyxl
import shutil
import os

# --- Phase 0: Copy workbook and inspect layout ---
os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')

wb = openpyxl.load_workbook('/root/output/result.xlsx')

# Print sheet names
print('Sheets:', wb.sheetnames)

task = wb['Task']
data = wb['Data']

# Inspect Task sheet: row 10 (year headers), column D (series codes), and key rows
print('\n--- Task sheet row 10 (headers/years) ---')
for col in range(1, 15):
    print(f'  col {col} ({openpyxl.utils.get_column_letter(col)}): {task.cell(row=10, column=col).value}')

print('\n--- Task sheet column D (series codes) rows 11-55 ---')
for row in range(11, 56):
    d_val = task.cell(row=row, column=4).value
    a_val = task.cell(row=row, column=1).value
    b_val = task.cell(row=row, column=2).value
    h_val = task.cell(row=row, column=8).value
    print(f'  row {row}: A={a_val}, B={b_val}, D={d_val}, H={h_val}')

print('\n--- Data sheet rows 19-40, columns A-Z ---')
for row in range(19, 41):
    vals = []
    for col in range(1, 27):
        v = data.cell(row=row, column=col).value
        if v is not None:
            vals.append(f'{openpyxl.utils.get_column_letter(col)}{row}={v}')
    print(f'  row {row}: {vals}')

print('\n--- Data sheet row 20 (possible header row) ---')
for col in range(1, 27):
    v = data.cell(row=20, column=col).value
    if v is not None:
        print(f'  col {openpyxl.utils.get_column_letter(col)}: {v}')

wb.close()
print('\nPhase 0 inspection complete.')
```

After reviewing the output, execute the following script. **Adapt the column letters and row numbers below if the Phase 0 inspection reveals different layout details.** The template below assumes the most common layout seen in this benchmark family (years in row 20 of Data starting at column B; series codes in column A of Data rows 21-38; values filling the grid). Adjust as needed.

```python
import openpyxl

wb = openpyxl.load_workbook('/root/output/result.xlsx')
task = wb['Task']
data = wb['Data']

# ---- Determine Data sheet layout from Phase 0 output ----
# We need:
#   data_year_row : the row in Data that contains year headers
#   data_code_col : the column letter in Data that contains series codes
#   data_first_row, data_last_row : row range 21:38
#   data_first_val_col, data_last_val_col : first and last value columns in Data
# These will be confirmed/adjusted from Phase 0 output.

# Read actual Data layout
# Find year header row (row 20) and value columns
data_year_row = 20
data_code_col = 'A'  # column with series codes
data_first_row = 21
data_last_row = 38

# Find the range of value columns in Data by scanning row 20 for numeric years
data_val_cols = []
for col in range(1, 50):
    v = data.cell(row=data_year_row, column=col).value
    if v is not None and isinstance(v, (int, float)) and 1900 < v < 2100:
        data_val_cols.append(col)
    elif isinstance(v, str) and v.isdigit() and 1900 < int(v) < 2100:
        data_val_cols.append(col)

if not data_val_cols:
    # Maybe years are in row 19 or another row; try scanning
    for try_row in [19, 20, 21]:
        for col in range(1, 50):
            v = data.cell(row=try_row, column=col).value
            if v is not None and isinstance(v, (int, float)) and 1900 < v < 2100:
                data_val_cols.append(col)
        if data_val_cols:
            data_year_row = try_row
            break

data_first_val_col_letter = openpyxl.utils.get_column_letter(min(data_val_cols)) if data_val_cols else 'B'
data_last_val_col_letter = openpyxl.utils.get_column_letter(max(data_val_cols)) if data_val_cols else 'S'

print(f'Data year row: {data_year_row}')
print(f'Data value columns: {data_first_val_col_letter} to {data_last_val_col_letter}')
print(f'Data code column: {data_code_col}')
print(f'Data rows: {data_first_row}:{data_last_row}')

# ---- Step 1: Populate H12:L17, H19:L24, H26:L31 with INDEX/MATCH/MATCH formulas ----
# Task sheet: years are in row 10, series codes in column D
# Formula pattern: =INDEX(Data!B21:S38, MATCH(D12,Data!A21:A38,0), MATCH(H10,Data!B20:S20,0))
# Adjust range letters from inspection.

block_rows = list(range(12, 18)) + list(range(19, 25)) + list(range(26, 32))
task_year_row = 10
task_code_col = 'D'  # column with series codes on Task sheet

# Build the data range references
data_range = f"Data!{data_first_val_col_letter}{data_first_row}:{data_last_val_col_letter}{data_last_row}"
code_range = f"Data!{data_code_col}{data_first_row}:{data_code_col}{data_last_row}"
year_range = f"Data!{data_first_val_col_letter}{data_year_row}:{data_last_val_col_letter}{data_year_row}"

print(f'Data range: {data_range}')
print(f'Code range: {code_range}')
print(f'Year range: {year_range}')

for row in block_rows:
    for col_idx in range(8, 13):  # H=8 to L=12
        col_letter = openpyxl.utils.get_column_letter(col_idx)
        year_cell = f'{col_letter}${task_year_row}'  # e.g. H$10
        code_cell = f'${task_code_col}{row}'  # e.g. $D12
        formula = f'=INDEX({data_range},MATCH({code_cell},{code_range},0),MATCH({year_cell},{year_range},0))'
        task.cell(row=row, column=col_idx).value = formula
        print(f'  {col_letter}{row}: {formula}')

# ---- Step 2: Net capacity headroom in H35:L40 ----
# Formula: (Available Care Slots - Occupied Care Slots) / Staffed Bed Capacity * 100
# Available Care Slots = rows 12:17, Occupied Care Slots = rows 19:24, Staffed Bed Capacity = rows 26:31
# Net headroom row i (0-5) -> (H(12+i) - H(19+i)) / H(26+i) * 100

for i in range(6):
    avail_row = 12 + i
    occup_row = 19 + i
    cap_row = 26 + i
    target_row = 35 + i
    for col_idx in range(8, 13):
        col_letter = openpyxl.utils.get_column_letter(col_idx)
        formula = f'=({col_letter}{avail_row}-{col_letter}{occup_row})/{col_letter}{cap_row}*100'
        task.cell(row=target_row, column=col_idx).value = formula
        print(f'  {col_letter}{target_row}: {formula}')

# ---- Step 2 continued: Summary stats in H42:L47 ----
# Row 42: MIN, 43: MAX, 44: MEDIAN, 45: AVERAGE, 46: 25th percentile, 47: 75th percentile
# These are column-wise over H35:H40 etc.

stat_formulas = [
    (42, 'MIN'),
    (43, 'MAX'),
    (44, 'MEDIAN'),
    (45, 'AVERAGE'),
    (46, 'PERCENTILE'),  # 25th
    (47, 'PERCENTILE'),  # 75th
]

for col_idx in range(8, 13):
    col_letter = openpyxl.utils.get_column_letter(col_idx)
    rng = f'{col_letter}35:{col_letter}40'
    for stat_row, func in stat_formulas:
        if func == 'PERCENTILE' and stat_row == 46:
            formula = f'=PERCENTILE({rng},0.25)'
        elif func == 'PERCENTILE' and stat_row == 47:
            formula = f'=PERCENTILE({rng},0.75)'
        else:
            formula = f'={func}({rng})'
        task.cell(row=stat_row, column=col_idx).value = formula
        print(f'  {col_letter}{stat_row}: {formula}')

# ---- Step 3: Weighted mean in H50:L50 ----
# =SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
for col_idx in range(8, 13):
    col_letter = openpyxl.utils.get_column_letter(col_idx)
    pct_range = f'{col_letter}35:{col_letter}40'
    wt_range = f'{col_letter}26:{col_letter}31'
    formula = f'=SUMPRODUCT({pct_range},{wt_range})/SUM({wt_range})'
    task.cell(row=50, column=col_idx).value = formula
    print(f'  {col_letter}50: {formula}')

# ---- Save ----
wb.save('/root/output/result.xlsx')
wb.close()
print('\nWorkbook saved to /root/output/result.xlsx')
```

After running Phase 0, review the output carefully. If the Data sheet layout differs from assumptions (e.g., year header row is not 20, series codes are not in column A, or the data rows are not 21:38), adjust the second script's variables accordingly before running it. Key things to verify from Phase 0:
- Which row in Data contains year headers
- Which column in Data contains series codes
- The exact row range for data values (should be 21:38 per the instructions)
- Which columns contain the year values
- The Task sheet's year row (should be row 10) and series code column (should be D)

Also verify after saving:
1. Open the saved file and confirm cells H12, H35, H42, and H50 contain formula strings (not None).
2. Confirm no extra sheets were added.

```python
import openpyxl
wb = openpyxl.load_workbook('/root/output/result.xlsx')
task = wb['Task']
print('Sheets:', wb.sheetnames)
print('H12:', task['H12'].value)
print('H19:', task['H19'].value)
print('H26:', task['H26'].value)
print('H35:', task['H35'].value)
print('H42:', task['H42'].value)
print('H46:', task['H46'].value)
print('H50:', task['H50'].value)
print('L17:', task['L17'].value)
print('L40:', task['L40'].value)
print('L47:', task['L47'].value)
print('L50:', task['L50'].value)
wb.close()
print('Verification complete.')
```

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=hard, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.