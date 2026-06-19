# Task Instruction

Execute the following Python script to produce /root/output/result.xlsx.

```python
import openpyxl, os, shutil

# --- Phase 0: Copy workbook and inspect layout ---
os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')

wb = openpyxl.load_workbook('/root/output/result.xlsx')
print('Sheets:', wb.sheetnames)

task = wb['Task']
data = wb['Data']

# Inspect Task sheet layout
print('\n--- Task sheet inspection ---')
for r in range(1, 55):
    vals = []
    for c in range(1, 15):
        v = task.cell(row=r, column=c).value
        if v is not None:
            vals.append(f"{openpyxl.utils.get_column_letter(c)}{r}={v}")
    if vals:
        print(f"Row {r}: {vals}")

# Inspect Data sheet layout
print('\n--- Data sheet inspection (rows 1-5 and 18-40) ---')
for r in list(range(1, 6)) + list(range(18, 42)):
    vals = []
    for c in range(1, 30):
        v = data.cell(row=r, column=c).value
        if v is not None:
            vals.append(f"{openpyxl.utils.get_column_letter(c)}{r}={v}")
    if vals:
        print(f"Row {r}: {vals}")

wb.close()
```

After inspecting the output, proceed with Phase 1 below. Adjust row/column references if the inspection reveals different layout than assumed.

```python
import openpyxl
from openpyxl.utils import get_column_letter

wb = openpyxl.load_workbook('/root/output/result.xlsx')
task = wb['Task']
data = wb['Data']

# Re-inspect key cells to confirm layout
# Row 10 of Task should have years in columns H-L
print('Task row 10 (years):', [task.cell(row=10, column=c).value for c in range(8, 13)])
# Column D of Task rows 12-17, 19-24, 26-31 should have series codes
for r in list(range(12, 18)) + list(range(19, 25)) + list(range(26, 32)):
    print(f"Task D{r} = {task.cell(row=r, column=4).value}")

# Identify Data sheet structure for rows 21-38
# Find which row has headers and which column has the series codes
print('\nData row 20 (likely header):', [data.cell(row=20, column=c).value for c in range(1, 30) if data.cell(row=20, column=c).value is not None])
print('Data row 21:', [(get_column_letter(c), data.cell(row=21, column=c).value) for c in range(1, 30) if data.cell(row=21, column=c).value is not None])

# Find the column in Data that contains series codes (likely column A or B)
# and the row that contains years
for r in range(1, 22):
    row_vals = [(get_column_letter(c), data.cell(row=r, column=c).value) for c in range(1, 30) if data.cell(row=r, column=c).value is not None]
    if row_vals:
        print(f"Data row {r}: {row_vals}")

wb.close()
```

After confirming the exact layout, execute Phase 2 — the formula-writing phase. The formulas below use INDEX/MATCH/MATCH pattern. Adjust the Data range references based on the actual inspection.

```python
import openpyxl
from openpyxl.utils import get_column_letter

wb = openpyxl.load_workbook('/root/output/result.xlsx')
task = wb['Task']

# ============================================================
# STEP 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas
# ============================================================
# Pattern: INDEX(Data!<data_range>, MATCH($D{row}, Data!<series_col>, 0), MATCH(H$10, Data!<year_row>, 0))
# Data rows 21:38 contain the records.
# We need to identify:
#   - data_range: the numeric values area on Data sheet (rows 21-38, columns with year data)
#   - series_col: the column on Data containing series codes (rows 21-38)
#   - year_row: the row on Data containing year headers
# These will be confirmed from Phase 0/1 inspection.
#
# IMPORTANT: Adjust the following based on actual inspection results.
# Typical layout: Data has series codes in column A rows 21-38,
# years in some row (e.g., row 20) starting from column B or C,
# and numeric data in the corresponding grid.

# --- After inspection, fill in the actual references ---
# Example (ADJUST BASED ON INSPECTION):
# Suppose Data has:
#   Series codes in column A, rows 21-38  -> Data!$A$21:$A$38
#   Years in row 20, columns B onwards    -> Data!$B$20:${last_col}$20  
#   Data values in B21:{last_col}38       -> Data!$B$21:${last_col}$38

# Read actual Data layout to determine ranges programmatically
data = wb['Data']

# Find the column containing series codes for rows 21-38
series_col_idx = None
for c in range(1, 30):
    v = data.cell(row=21, column=c).value
    if v is not None and isinstance(v, str) and not isinstance(v, (int, float)):
        # Check if this looks like a series code
        series_col_idx = c
        break

print(f"Series code column index: {series_col_idx}")
series_col_letter = get_column_letter(series_col_idx) if series_col_idx else 'A'

# Find the year header row (should be row 20 or nearby)
# Look for a row just above 21 that contains year values matching Task row 10
task_years = [task.cell(row=10, column=c).value for c in range(8, 13)]
print(f"Task years: {task_years}")

year_row = None
data_col_start = None
data_col_end = None
for r in range(15, 21):
    for c in range(1, 30):
        v = data.cell(row=r, column=c).value
        if v is not None and v == task_years[0]:
            year_row = r
            data_col_start = c
            # Find end
            for c2 in range(c, 50):
                if data.cell(row=r, column=c2).value is None:
                    data_col_end = c2 - 1
                    break
            break
    if year_row:
        break

print(f"Year row: {year_row}, data columns: {data_col_start}-{data_col_end}")
start_letter = get_column_letter(data_col_start)
end_letter = get_column_letter(data_col_end)

# Build the INDEX/MATCH formulas
# INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
data_range = f"Data!${start_letter}$21:${end_letter}$38"
series_range = f"Data!${series_col_letter}$21:${series_col_letter}$38"
year_range = f"Data!${start_letter}${year_row}:${end_letter}${year_row}"

print(f"Data range: {data_range}")
print(f"Series range: {series_range}")
print(f"Year range: {year_range}")

# Populate the three blocks
blocks = [
    (12, 17),  # H12:L17
    (19, 24),  # H19:L24
    (26, 31),  # H26:L31
]

for (row_start, row_end) in blocks:
    for row in range(row_start, row_end + 1):
        for col in range(8, 13):  # H=8 through L=12
            col_letter = get_column_letter(col)
            formula = f'=INDEX({data_range},MATCH($D{row},{series_range},0),MATCH({col_letter}$10,{year_range},0))'
            task.cell(row=row, column=col).value = formula
            print(f"  {col_letter}{row}: {formula}")

# ============================================================
# STEP 2: Net container flow in H35:L40
# ============================================================
# Formula: (Loaded Containers Inbound - Loaded Containers Outbound) / Terminal Throughput Capacity * 100
# Block 1 (rows 12-17): first metric
# Block 2 (rows 19-24): second metric  
# Block 3 (rows 26-31): third metric
# We need to identify which block is Inbound, Outbound, and Capacity.
# Check the labels in column B or C for rows 11, 18, 25
print('\nBlock labels:')
for r in [11, 18, 25, 34]:
    for c in range(1, 8):
        v = task.cell(row=r, column=c).value
        if v is not None:
            print(f"  {get_column_letter(c)}{r}: {v}")

# Also check row 35 area labels
for r in range(33, 48):
    for c in range(1, 8):
        v = task.cell(row=r, column=c).value
        if v is not None:
            print(f"  {get_column_letter(c)}{r}: {v}")

# Based on the typical pattern:
# Block 1 (12-17) = Loaded Containers Inbound
# Block 2 (19-24) = Loaded Containers Outbound  
# Block 3 (26-31) = Terminal Throughput Capacity
# Net flow = (Block1 - Block2) / Block3 * 100

# Rows 35-40 correspond to the 6 ports (same order as rows 12-17, 19-24, 26-31)
for i in range(6):  # 6 ports
    inbound_row = 12 + i
    outbound_row = 19 + i
    capacity_row = 26 + i
    result_row = 35 + i
    for col in range(8, 13):  # H through L
        col_letter = get_column_letter(col)
        formula = f'=({col_letter}{inbound_row}-{col_letter}{outbound_row})/{col_letter}{capacity_row}*100'
        task.cell(row=result_row, column=col).value = formula
        print(f"  {col_letter}{result_row}: {formula}")

# ============================================================
# STEP 2 continued: Summary statistics in H42:L47
# ============================================================
# Row 42: MIN, Row 43: MAX, Row 44: MEDIAN, Row 45: AVERAGE, Row 46: 25th percentile, Row 47: 75th percentile
# Check actual labels
print('\nSummary stat labels:')
for r in range(41, 52):
    for c in range(1, 8):
        v = task.cell(row=r, column=c).value
        if v is not None:
            print(f"  {get_column_letter(c)}{r}: {v}")

# Apply summary formulas (adjust row mapping based on label inspection)
stat_formulas = {
    42: 'MIN',
    43: 'MAX',
    44: 'MEDIAN',
    45: 'AVERAGE',  # simple mean
    46: 'PERCENTILE',  # 25th - use PERCENTILE not PERCENTILE.INC
    47: 'PERCENTILE',  # 75th
}

for col in range(8, 13):  # H through L
    col_letter = get_column_letter(col)
    data_ref = f'{col_letter}35:{col_letter}40'
    
    for row, func in stat_formulas.items():
        if func == 'PERCENTILE' and row == 46:
            formula = f'=PERCENTILE({data_ref},0.25)'
        elif func == 'PERCENTILE' and row == 47:
            formula = f'=PERCENTILE({data_ref},0.75)'
        else:
            formula = f'={func}({data_ref})'
        task.cell(row=row, column=col).value = formula
        print(f"  {col_letter}{row}: {formula}")

# ============================================================
# STEP 3: Weighted mean in H50:L50 using SUMPRODUCT
# ============================================================
# Weighted mean = SUMPRODUCT(net_flow_values, capacity_weights) / SUM(capacity_weights)
for col in range(8, 13):  # H through L
    col_letter = get_column_letter(col)
    net_ref = f'{col_letter}35:{col_letter}40'
    cap_ref = f'{col_letter}26:{col_letter}31'
    formula = f'=SUMPRODUCT({net_ref},{cap_ref})/SUM({cap_ref})'
    task.cell(row=50, column=col).value = formula
    print(f"  {col_letter}50: {formula}")

# Save
wb.save('/root/output/result.xlsx')
print('\nWorkbook saved to /root/output/result.xlsx')
wb.close()
```

IMPORTANT EXECUTION NOTES:
1. Run Phase 0 and Phase 1 (inspection) FIRST. Read all printed output carefully.
2. Before running Phase 2 (formula writing), verify:
   - Which column in Data sheet contains series codes (adjust `series_col_idx` logic if needed)
   - Which row in Data sheet contains year headers (adjust `year_row` search if needed)
   - Which blocks correspond to Inbound, Outbound, and Capacity (check the labels printed)
   - Which rows correspond to MIN, MAX, MEDIAN, AVERAGE, 25th, 75th percentile (check labels)
   - The stat row assignments (42-47) match the actual label positions
3. If the label inspection reveals different ordering (e.g., MAX before MIN, or blocks in different order), adjust the formulas accordingly BEFORE writing them.
4. Use 'PERCENTILE' (not 'PERCENTILE.INC' or 'PERCENTILE.EXC') to avoid #NAME? errors.
5. After saving, re-open and spot-check a few cells to confirm formulas were written correctly.
6. Do NOT add any new sheets, macros, or VBA code. Only modify existing cells in the Task sheet.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=medium, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.