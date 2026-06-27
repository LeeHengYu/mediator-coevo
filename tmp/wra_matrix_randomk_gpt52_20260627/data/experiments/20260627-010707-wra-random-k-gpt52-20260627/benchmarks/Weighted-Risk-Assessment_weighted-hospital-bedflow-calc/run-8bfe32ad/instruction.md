# Task Instruction

Execute the following steps exactly in order.

## 0. Inspect the workbook layout

```python
import openpyxl, shutil, os

wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
ws_task = wb['Task']
ws_data = wb['Data']

print('=== Task sheet layout (rows 1-55, cols A-M) ===')
for row in ws_task.iter_rows(min_row=1, max_row=55, min_col=1, max_col=13, values_only=False):
    for cell in row:
        if cell.value is not None:
            print(f'  {cell.coordinate}: {repr(cell.value)}')

print('\n=== Data sheet layout (rows 1-5 header area) ===')
for row in ws_data.iter_rows(min_row=1, max_row=5, min_col=1, max_col=20, values_only=False):
    for cell in row:
        if cell.value is not None:
            print(f'  {cell.coordinate}: {repr(cell.value)}')

print('\n=== Data sheet rows 18-40 ===')
for row in ws_data.iter_rows(min_row=18, max_row=40, min_col=1, max_col=20, values_only=False):
    for cell in row:
        if cell.value is not None:
            print(f'  {cell.coordinate}: {repr(cell.value)}')

wb.close()
```

Read the output carefully. Identify:
- The exact column letters/numbers where series codes live in Task!D (rows 12-17, 19-24, 26-31).
- The exact row 10 cells that contain year headers in Task!H10:L10.
- The Data sheet layout: which row contains headers, which column has series codes, which rows/columns have the actual data for rows 21-38.
- The exact labels and layout for rows 35-47 and row 50 on the Task sheet.

## 1. Write a Python script that populates the formulas

Create `/root/solve.py` with the following logic. **Adapt cell references based on what you discovered in step 0.**

```python
import openpyxl, os, shutil

# Load workbook (keep formatting)
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
ws = wb['Task']

# ---- Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31 ----
# For each block, each row has a series code in column D.
# Each column H-L corresponds to a year in row 10.
# Use INDEX/MATCH pattern against Data!$A$21:$A$38 (series codes) and Data!$A$20:$XX$20 (year headers)
# Adjust the Data ranges based on inspection.
#
# Formula pattern per cell (row r, col c where H=8, I=9, ..., L=12):
#   =INDEX(Data!<data_range>, MATCH($D{r}, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
#
# IMPORTANT: The Data range for INDEX should cover the data area (rows 21-38, data columns).
# The MATCH for series code should search the series code column of Data rows 21-38.
# The MATCH for year should search the year header row of Data.

blocks = [
    (12, 17),  # H12:L17
    (19, 24),  # H19:L24
    (26, 31),  # H26:L31
]

# YOU MUST ADAPT THESE REFERENCES after inspecting the Data sheet:
# data_series_col = the column letter in Data that has series codes (e.g., 'A' or 'B')
# data_year_row = the row number in Data that has year headers
# data_top_left / data_bottom_right = the rectangular data area
# Example (adapt!):
# =INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))

for (start_row, end_row) in blocks:
    for r in range(start_row, end_row + 1):
        for c in range(8, 13):  # columns H(8) through L(12)
            col_letter = openpyxl.utils.get_column_letter(c)
            # Build the formula - adapt Data references!
            formula = (
                f'=INDEX(Data!$B$21:$Z$38,'
                f'MATCH($D{r},Data!$A$21:$A$38,0),'
                f'MATCH({col_letter}$10,Data!$B$20:$Z$20,0))'
            )
            ws.cell(row=r, column=c).value = formula

# ---- Step 2: Net patient flow in H35:L40 ----
# Formula: (Admissions - Discharges) / Effective Bed Capacity * 100
# Admissions are in rows 12-17, Discharges in rows 19-24, Capacity in rows 26-31
# Row 35 corresponds to row 12 (hospital 1), row 36 to row 13, etc.

for i in range(6):  # 0..5 for six hospitals
    adm_row = 12 + i
    dis_row = 19 + i
    cap_row = 26 + i
    flow_row = 35 + i
    for c in range(8, 13):
        col_letter = openpyxl.utils.get_column_letter(c)
        formula = f'=({col_letter}{adm_row}-{col_letter}{dis_row})/{col_letter}{cap_row}*100'
        ws.cell(row=flow_row, column=c).value = formula

# ---- Statistics in H42:L47 ----
# Row 42: MIN, 43: MAX, 44: MEDIAN, 45: AVERAGE, 46: 25th percentile, 47: 75th percentile
# All column-wise over H35:H40 etc.

stat_funcs = [
    (42, 'MIN({col}35:{col}40)'),
    (43, 'MAX({col}35:{col}40)'),
    (44, 'MEDIAN({col}35:{col}40)'),
    (45, 'AVERAGE({col}35:{col}40)'),
    (46, 'PERCENTILE({col}35:{col}40,0.25)'),
    (47, 'PERCENTILE({col}35:{col}40,0.75)'),
]

for (stat_row, tmpl) in stat_funcs:
    for c in range(8, 13):
        col_letter = openpyxl.utils.get_column_letter(c)
        formula = '=' + tmpl.format(col=col_letter)
        ws.cell(row=stat_row, column=c).value = formula

# ---- Step 3: Weighted mean in H50:L50 ----
# =SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
# Actually the instruction says use SUMPRODUCT with percentages as values and capacity as weights.
# Weighted mean = SUMPRODUCT(values, weights) / SUM(weights)

for c in range(8, 13):
    col_letter = openpyxl.utils.get_column_letter(c)
    formula = f'=SUMPRODUCT({col_letter}35:{col_letter}40,{col_letter}26:{col_letter}31)/SUM({col_letter}26:{col_letter}31)'
    ws.cell(row=50, column=c).value = formula

# ---- Save ----
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
wb.close()
print('Saved to /root/output/result.xlsx')
```

## 2. Adapt the script based on inspection

After running step 0, you MUST update the Data sheet references in the script:
- Replace `Data!$A$21:$A$38` with the actual column and row range containing series codes.
- Replace `Data!$B$21:$Z$38` with the actual rectangular data area.
- Replace `Data!$B$20:$Z$20` with the actual year header row range.
- Check whether the statistics row assignments (42-47) match the labels you see on the Task sheet. The order might be different (e.g., min/max/median/mean/percentiles). Match each formula to the label in column D or wherever the stat labels are.
- Check whether the flow rows (35-40) and weighted mean row (50) match the actual layout.

## 3. Run the script

```bash
cd /root && python solve.py
```

## 4. Verify the output

```python
import openpyxl
wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb['Task']

print('=== Lookup block H12:L17 ===')
for r in range(12, 18):
    for c in range(8, 13):
        print(f'  {ws.cell(r,c).coordinate}: {ws.cell(r,c).value}')

print('\n=== Flow block H35:L40 ===')
for r in range(35, 41):
    for c in range(8, 13):
        print(f'  {ws.cell(r,c).coordinate}: {ws.cell(r,c).value}')

print('\n=== Stats H42:L47 ===')
for r in range(42, 48):
    for c in range(8, 13):
        print(f'  {ws.cell(r,c).coordinate}: {ws.cell(r,c).value}')

print('\n=== Weighted mean H50:L50 ===')
for c in range(8, 13):
    print(f'  {ws.cell(50,c).coordinate}: {ws.cell(50,c).value}')

wb.close()
```

Confirm that every target cell contains a formula string (starting with '='). None of them should be `None`.

## 5. Run the verifier if available

```bash
cd /root && python -m pytest tests/ -v 2>&1 | head -80
```

If tests fail, read the error messages carefully, fix the formulas in solve.py, re-run, and re-verify. Common issues:
- Wrong Data sheet ranges (off-by-one in rows/columns)
- PERCENTILE function might need to be PERCENTILE.INC in some Excel versions (try PERCENTILE first since the avoid-artifact warns about #NAME? errors from wrong function names)
- Statistics rows might not be in the order assumed above
- The lookup formula pattern might need adjustment

**Critical**: Do NOT skip the initial inspection step. The previous failure happened because formulas were written to wrong cells or not written at all. Inspect first, adapt, then write.

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