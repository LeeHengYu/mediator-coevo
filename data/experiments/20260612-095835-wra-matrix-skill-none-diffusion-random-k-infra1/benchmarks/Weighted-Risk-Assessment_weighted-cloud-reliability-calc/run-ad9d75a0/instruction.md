# Task Instruction

Execute the following Python script to read the workbook, populate formulas, and save the result.

```python
import openpyxl
import os
import shutil

# Copy workbook to output
os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')

wb = openpyxl.load_workbook('/root/output/result.xlsx')

# ── Inspection: understand the layout ──
ws_task = wb['Task']
ws_data = wb['Data']

print('=== Task sheet row 10 (years) ===')
for col in range(1, 15):
    print(f'  col {col} ({openpyxl.utils.get_column_letter(col)}): {ws_task.cell(row=10, column=col).value}')

print('\n=== Task sheet column D (series codes) ===')
for row in range(11, 55):
    val = ws_task.cell(row=row, column=4).value
    if val is not None:
        print(f'  row {row}: {val}')

print('\n=== Task sheet column A-G rows 11-50 (labels/structure) ===')
for row in range(11, 55):
    vals = [ws_task.cell(row=row, column=c).value for c in range(1, 8)]
    if any(v is not None for v in vals):
        print(f'  row {row}: {vals}')

print('\n=== Data sheet rows 19-40 (header + data) ===')
for row in range(19, 41):
    vals = [ws_data.cell(row=row, column=c).value for c in range(1, 25)]
    if any(v is not None for v in vals):
        print(f'  row {row}: {vals}')

print('\n=== Data sheet row 20 (possible header) ===')
for col in range(1, 25):
    v = ws_data.cell(row=20, column=col).value
    if v is not None:
        print(f'  col {col} ({openpyxl.utils.get_column_letter(col)}): {v}')

print('\n=== Task sheet rows 35-50 ===')
for row in range(35, 51):
    vals = [ws_task.cell(row=row, column=c).value for c in range(1, 13)]
    if any(v is not None for v in vals):
        print(f'  row {row}: {vals}')

print('\n=== Task sheet H42:L47 labels ===')
for row in range(42, 48):
    vals = [ws_task.cell(row=row, column=c).value for c in range(1, 8)]
    print(f'  row {row}: {vals}')

print('\n=== Task sheet row 50 ===')
vals = [ws_task.cell(row=50, column=c).value for c in range(1, 13)]
print(f'  row 50: {vals}')

wb.close()
```

After inspecting the output, run the following script that populates formulas. Adjust column letters and row numbers based on what the inspection reveals, but the expected layout is:

- Row 10 columns H-L contain years.
- Column D rows 12-17 contain series codes for block 1 (e.g., Successful API Requests).
- Column D rows 19-24 contain series codes for block 2 (e.g., Failed API Requests).
- Column D rows 26-31 contain series codes for block 3 (e.g., Compute Capacity).
- Data sheet rows 21-38 contain the lookup table with series codes in one column and year values across columns.

Here is the formula-population script (adapt after inspection if needed):

```python
import openpyxl
import os

wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws_task = wb['Task']
ws_data = wb['Data']

# Identify Data sheet structure
# Find the header row for the data table (row 20 or 21)
# We need to know which column has series codes and which row has years
# Print data sheet column A rows 20-22 to confirm
print('Data col A rows 20-22:')
for r in range(20, 23):
    print(f'  row {r}: {ws_data.cell(row=r, column=1).value}')

# Print data sheet row 20 to find year headers
print('Data row 20 all:')
for c in range(1, 25):
    v = ws_data.cell(row=20, column=c).value
    if v is not None:
        print(f'  {openpyxl.utils.get_column_letter(c)}{20}: {v}')

wb.close()
```

Then, once the exact layout is confirmed, run this final script:

```python
import openpyxl
from openpyxl.utils import get_column_letter
import os

wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb['Task']

# ── Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31 ──
# Pattern: INDEX(MATCH) using Data sheet
# Series code is in column D of Task sheet, year is in row 10 of Task sheet
# Data!$A$21:$A$38 has series codes, Data row 20 has year headers
# Data!$A$20:$XX$38 is the table

# We need to know the last column of Data. Let's find it.
ws_data = wb['Data']
data_last_col = 1
for c in range(1, 100):
    if ws_data.cell(row=20, column=c).value is not None:
        data_last_col = c
data_last_col_letter = get_column_letter(data_last_col)
print(f'Data last col: {data_last_col_letter}{data_last_col}')

# The lookup formula for each cell (row r, col c) in Task sheet:
# =INDEX(Data!$A$21:$XX$38, MATCH($D{r}, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$A$20:$XX$20, 0))
# where H$10 is the year cell (column changes with c)

# Build the formula
# Data range: Data!$A$20:${data_last_col_letter}$38
# Series codes: Data!$A$21:$A$38
# Year headers: Data!$A$20:${data_last_col_letter}$20
# Data body: Data!$A$21:${data_last_col_letter}$38

blocks = [
    (12, 17),  # H12:L17
    (19, 24),  # H19:L24
    (26, 31),  # H26:L31
]

for start_row, end_row in blocks:
    for r in range(start_row, end_row + 1):
        for c_idx in range(8, 13):  # H=8, I=9, J=10, K=11, L=12
            col_letter = get_column_letter(c_idx)
            formula = (
                f'=INDEX(Data!$A$21:${data_last_col_letter}$38,'
                f'MATCH($D{r},Data!$A$21:$A$38,0),'
                f'MATCH({col_letter}$10,Data!$A$20:${data_last_col_letter}$20,0))'
            )
            ws.cell(row=r, column=c_idx, value=formula)

print('Step 1 formulas written.')

# ── Step 2: Net reliability gap in H35:L40 ──
# (Successful API Requests - Failed API Requests) / Compute Capacity * 100
# Block 1 (Success): rows 12-17
# Block 2 (Failure): rows 19-24
# Block 3 (Capacity): rows 26-31
# Net reliability gap rows 35-40 map to regions in order (6 regions)

for i in range(6):
    success_row = 12 + i
    failure_row = 19 + i
    capacity_row = 26 + i
    target_row = 35 + i
    for c_idx in range(8, 13):
        col_letter = get_column_letter(c_idx)
        formula = (
            f'=({col_letter}{success_row}-{col_letter}{failure_row})'
            f'/{col_letter}{capacity_row}*100'
        )
        ws.cell(row=target_row, column=c_idx, value=formula)

print('Step 2 net reliability gap written.')

# ── Step 2 continued: Stats in H42:L47 ──
# Row 42: MIN, 43: MAX, 44: MEDIAN, 45: AVERAGE, 46: PERCENTILE 25th, 47: PERCENTILE 75th
# Check what labels are in column A-G for rows 42-47
for r in range(42, 48):
    label = ws.cell(row=r, column=4).value or ws.cell(row=r, column=3).value or ws.cell(row=r, column=2).value or ws.cell(row=r, column=1).value
    print(f'  row {r} label: {label}')

# Map labels to functions - we'll read them and assign accordingly
# Expected order based on instruction: min, max, median, mean, 25th percentile, 75th percentile
stat_formulas = {}
for r in range(42, 48):
    # Collect all text in columns A-G
    label = ''
    for cc in range(1, 8):
        v = ws.cell(row=r, column=cc).value
        if v is not None:
            label = str(v).strip().lower()
    stat_formulas[r] = label

print(f'Stat labels: {stat_formulas}')

for r in range(42, 48):
    label = stat_formulas[r]
    for c_idx in range(8, 13):
        col_letter = get_column_letter(c_idx)
        rng = f'{col_letter}35:{col_letter}40'
        if 'min' in label:
            formula = f'=MIN({rng})'
        elif 'max' in label:
            formula = f'=MAX({rng})'
        elif 'median' in label:
            formula = f'=MEDIAN({rng})'
        elif 'mean' in label or 'average' in label:
            formula = f'=AVERAGE({rng})'
        elif '25' in label or 'q1' in label or 'first' in label:
            formula = f'=PERCENTILE({rng},0.25)'
        elif '75' in label or 'q3' in label or 'third' in label:
            formula = f'=PERCENTILE({rng},0.75)'
        else:
            # Fallback: use instruction order
            idx = r - 42
            funcs = [
                f'=MIN({rng})', f'=MAX({rng})', f'=MEDIAN({rng})',
                f'=AVERAGE({rng})', f'=PERCENTILE({rng},0.25)', f'=PERCENTILE({rng},0.75)'
            ]
            formula = funcs[idx]
        ws.cell(row=r, column=c_idx, value=formula)

print('Step 2 stats written.')

# ── Step 3: Weighted mean in H50:L50 ──
# SUMPRODUCT(net_reliability_gap, compute_capacity) / SUM(compute_capacity)
for c_idx in range(8, 13):
    col_letter = get_column_letter(c_idx)
    formula = (
        f'=SUMPRODUCT({col_letter}35:{col_letter}40,{col_letter}26:{col_letter}31)'
        f'/SUM({col_letter}26:{col_letter}31)'
    )
    ws.cell(row=50, column=c_idx, value=formula)

print('Step 3 weighted mean written.')

# ── Verify all target cells have formulas ──
all_ok = True
target_ranges = [
    (12, 17), (19, 24), (26, 31),  # Step 1
    (35, 40), (42, 47),  # Step 2
    (50, 50),  # Step 3
]
for start_r, end_r in target_ranges:
    for r in range(start_r, end_r + 1):
        for c_idx in range(8, 13):
            v = ws.cell(row=r, column=c_idx).value
            if v is None or (isinstance(v, str) and not v.startswith('=')):
                print(f'WARNING: {get_column_letter(c_idx)}{r} = {v}')
                all_ok = False

if all_ok:
    print('All target cells contain formulas.')
else:
    print('Some cells are missing formulas!')

wb.save('/root/output/result.xlsx')
print('Saved to /root/output/result.xlsx')
```

IMPORTANT NOTES:
1. Run the inspection script FIRST. If the Data sheet layout differs from expectations (e.g., series codes are not in column A, or the year header row is not row 20), adjust the formula-population script accordingly before running it.
2. The stat labels in rows 42-47 must be matched to the correct functions. The script reads labels and maps them. If labels don't match, fall back to the instruction order: min, max, median, mean, 25th percentile, 75th percentile.
3. After saving, verify the file exists and has non-zero size.
4. Do NOT add any new sheets, macros, VBA, or external links.
5. The failed artifact from weighted-hospital-bedflow-calc shows that cells returning None is a common failure mode - ensure every target cell gets a formula string starting with '='.
6. If the inspection reveals that the Data sheet header row or series code column differs, you MUST adapt the INDEX/MATCH references in the formulas before writing them.

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