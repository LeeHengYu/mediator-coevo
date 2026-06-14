# Task Instruction

Execute the following Python script in a single step to inspect the workbook, populate all required formulas, and save the result.

```python
import openpyxl, os, json

# ── Phase 0: Inspect the workbook ──────────────────────────────────────────
src = '/root/data/workbook.xlsx'
wb = openpyxl.load_workbook(src)

# Inspect Data sheet structure
data_ws = wb['Data']
print('=== Data sheet rows 19-40 ===')
for r in range(19, 41):
    row_vals = []
    for c in range(1, 20):
        v = data_ws.cell(row=r, column=c).value
        row_vals.append(v)
    print(f'Row {r}: {row_vals}')

print('\n=== Data sheet row 1-5 (headers) ===')
for r in range(1, 6):
    row_vals = []
    for c in range(1, 20):
        v = data_ws.cell(row=r, column=c).value
        row_vals.append(v)
    print(f'Row {r}: {row_vals}')

# Inspect Task sheet structure
task_ws = wb['Task']
print('\n=== Task sheet rows 1-55 ===')
for r in range(1, 56):
    row_vals = []
    for c in range(1, 15):  # A-N
        v = task_ws.cell(row=r, column=c).value
        row_vals.append(v)
    print(f'Row {r}: {row_vals}')

# Print column letters for reference
from openpyxl.utils import get_column_letter
for c in range(1, 15):
    print(f'Col {c} = {get_column_letter(c)}')

wb.close()
```

After inspecting the output, run the following script (adjust Data sheet row/column references based on inspection if needed):

```python
import openpyxl, os
from openpyxl.utils import get_column_letter

src = '/root/data/workbook.xlsx'
wb = openpyxl.load_workbook(src)
task = wb['Task']

# ── Phase 1: Understand layout from inspection ────────────────────────────
# Task sheet layout (expected from inspection):
#   Row 10: year headers in H10:L10
#   Column D: series codes
#   H12:L17 = block 1 (e.g., Committed Funding)
#   H19:L24 = block 2 (e.g., Operating Spend)
#   H26:L31 = block 3 (e.g., Approved Budget Base)
#   H35:L40 = Net budget buffer
#   H42:L47 = Summary stats (min, max, median, mean, 25th pct, 75th pct)
#   H50:L50 = Weighted mean
#
# Data sheet: rows 21:38 contain source data.
# We need to figure out the Data sheet table structure to build INDEX/MATCH formulas.

# First, re-read to confirm exact layout
data_ws = wb['Data']

# Find the extent of Data rows 21:38
print('Re-checking Data rows 20-38 columns A-S')
for r in range(20, 39):
    vals = []
    for c in range(1, 20):
        vals.append(data_ws.cell(row=r, column=c).value)
    print(f'  Row {r}: {vals}')

# Find extent of data columns
max_data_col = 1
for c in range(1, 50):
    if data_ws.cell(row=21, column=c).value is not None:
        max_data_col = c
print(f'Max data col with value in row 21: {max_data_col} = {get_column_letter(max_data_col)}')

# Check what row has the series codes in Data and what row has years
# Typically row 20 or a header row has years, and column A or B has series codes
for r in range(1, 22):
    vals = [data_ws.cell(row=r, column=c).value for c in range(1, 20)]
    print(f'  Data Row {r}: {vals}')

wb.close()
```

After confirming the Data sheet layout, run the final script. The script below uses a generic approach — adjust the Data sheet references based on what you found in inspection:

```python
import openpyxl, os
from openpyxl.utils import get_column_letter
from copy import copy

src = '/root/data/workbook.xlsx'
wb = openpyxl.load_workbook(src)
task = wb['Task']

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTANT: Replace the following references with actual values from inspection.
# These are common patterns; adjust row/col numbers to match the real workbook.
# ══════════════════════════════════════════════════════════════════════════════

# Data sheet range containing source records (rows 21:38)
# Assume: Column A has series codes, Row 20 has year headers
# Data occupies Data!A20:<lastcol>38
# We'll build INDEX/MATCH formulas:
#   =INDEX(Data!$B$21:$<lastcol>$38, MATCH($D<row>, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$<lastcol>$20, 0))
#
# Adjust these after inspection!

# --- Read actual layout to set variables ---
data_ws = wb['Data']

# Find the row that contains year headers (likely row 20)
# and the column that contains series codes (likely column A=1)
# Detect: look for numeric year values in row 20
year_header_row = None
for candidate_row in [20, 19, 21]:
    v = data_ws.cell(row=candidate_row, column=2).value
    if v is not None and (isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit())):
        year_header_row = candidate_row
        break
if year_header_row is None:
    # Try checking what's in the Task sheet row 10 to find year values, then search Data
    sample_year = task.cell(row=10, column=8).value  # H10
    print(f'Sample year from Task H10: {sample_year}')
    for candidate_row in range(1, 40):
        for c in range(1, 20):
            if data_ws.cell(row=candidate_row, column=c).value == sample_year:
                year_header_row = candidate_row
                break
        if year_header_row:
            break

print(f'Year header row in Data: {year_header_row}')

# Find series code column (the column before the first year)
series_col = 1  # Usually column A
# Find the first and last data columns (years)
first_data_col = 2
last_data_col = 2
for c in range(2, 50):
    if data_ws.cell(row=year_header_row, column=c).value is not None:
        last_data_col = c
    else:
        break

print(f'Data columns: series={get_column_letter(series_col)}, years={get_column_letter(first_data_col)}:{get_column_letter(last_data_col)}')

# Data rows
data_first_row = 21
data_last_row = 38

# Build absolute references for Data sheet
series_range = f"Data!${get_column_letter(series_col)}${data_first_row}:${get_column_letter(series_col)}${data_last_row}"
year_range = f"Data!${get_column_letter(first_data_col)}${year_header_row}:${get_column_letter(last_data_col)}${year_header_row}"
data_range = f"Data!${get_column_letter(first_data_col)}${data_first_row}:${get_column_letter(last_data_col)}${data_last_row}"

print(f'Series range: {series_range}')
print(f'Year range: {year_range}')
print(f'Data range: {data_range}')

# ── Phase 2: Populate lookup blocks (H12:L17, H19:L24, H26:L31) ──────────
# Each cell: =INDEX(data_range, MATCH($D<row>, series_range, 0), MATCH(<col>$10, year_range, 0))

blocks = [
    (12, 17),  # rows 12-17
    (19, 24),  # rows 19-24
    (26, 31),  # rows 26-31
]

for (start_row, end_row) in blocks:
    for r in range(start_row, end_row + 1):
        for c in range(8, 13):  # H=8 to L=12
            col_letter = get_column_letter(c)
            formula = f'=INDEX({data_range},MATCH($D{r},{series_range},0),MATCH({col_letter}$10,{year_range},0))'
            task.cell(row=r, column=c).value = formula
            print(f'  {col_letter}{r}: {formula}')

# ── Phase 3: Net budget buffer (H35:L40) ─────────────────────────────────
# Formula: (Committed Funding - Operating Spend) / Approved Budget Base * 100
# Block 1 (H12:L17) = Committed Funding
# Block 2 (H19:L24) = Operating Spend  
# Block 3 (H26:L31) = Approved Budget Base
# Net buffer row i maps to department i:
#   Row 35 -> dept 1 (rows 12, 19, 26)
#   Row 36 -> dept 2 (rows 13, 20, 27)
#   etc.

for i in range(6):
    r_out = 35 + i
    r_committed = 12 + i
    r_operating = 19 + i
    r_budget = 26 + i
    for c in range(8, 13):  # H-L
        col_letter = get_column_letter(c)
        formula = f'=({col_letter}{r_committed}-{col_letter}{r_operating})/{col_letter}{r_budget}*100'
        task.cell(row=r_out, column=c).value = formula
        print(f'  {col_letter}{r_out}: {formula}')

# ── Phase 4: Summary statistics (H42:L47) ────────────────────────────────
# Row 42: MIN
# Row 43: MAX  
# Row 44: MEDIAN
# Row 45: AVERAGE (simple mean)
# Row 46: 25th percentile  (use PERCENTILE, NOT PERCENTILE.INC)
# Row 47: 75th percentile  (use PERCENTILE, NOT PERCENTILE.INC)
#
# CRITICAL: Use PERCENTILE (legacy name) to avoid #NAME? errors.

# First, verify the labels in column D or similar to confirm the order
for r in range(42, 48):
    label = task.cell(row=r, column=4).value  # column D
    if label is None:
        label = task.cell(row=r, column=3).value  # column C
    if label is None:
        label = task.cell(row=r, column=2).value  # column B
    print(f'  Row {r} label: {label}')

# Assign stats formulas - adjust row order if labels differ
stat_formulas = {
    42: 'MIN',
    43: 'MAX',
    44: 'MEDIAN',
    45: 'AVERAGE',
}

for r, func in stat_formulas.items():
    for c in range(8, 13):
        col_letter = get_column_letter(c)
        rng = f'{col_letter}35:{col_letter}40'
        formula = f'={func}({rng})'
        task.cell(row=r, column=c).value = formula
        print(f'  {col_letter}{r}: {formula}')

# Percentiles - use PERCENTILE (not PERCENTILE.INC or PERCENTILE.EXC)
for c in range(8, 13):
    col_letter = get_column_letter(c)
    rng = f'{col_letter}35:{col_letter}40'
    # 25th percentile
    formula_25 = f'=PERCENTILE({rng},0.25)'
    task.cell(row=46, column=c).value = formula_25
    print(f'  {col_letter}46: {formula_25}')
    # 75th percentile
    formula_75 = f'=PERCENTILE({rng},0.75)'
    task.cell(row=47, column=c).value = formula_75
    print(f'  {col_letter}47: {formula_75}')

# ── Phase 5: Weighted mean (H50:L50) ─────────────────────────────────────
# =SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
# Using SUMPRODUCT for weighted average:
# weighted mean = SUMPRODUCT(values, weights) / SUM(weights)

for c in range(8, 13):
    col_letter = get_column_letter(c)
    val_rng = f'{col_letter}35:{col_letter}40'
    wt_rng = f'{col_letter}26:{col_letter}31'
    formula = f'=SUMPRODUCT({val_rng},{wt_rng})/SUM({wt_rng})'
    task.cell(row=50, column=c).value = formula
    print(f'  {col_letter}50: {formula}')

# ── Phase 6: Save ─────────────────────────────────────────────────────────
os.makedirs('/root/output', exist_ok=True)
out = '/root/output/result.xlsx'
wb.save(out)
wb.close()
print(f'\nSaved to {out}')

# Verify by reopening
wb2 = openpyxl.load_workbook(out)
t2 = wb2['Task']
print('\n=== Verification: sample cells ===')
for r in [12, 19, 26, 35, 42, 46, 47, 50]:
    v = t2.cell(row=r, column=8).value  # H column
    print(f'  H{r}: {v}')
wb2.close()
print('Done.')
```

IMPORTANT EXECUTION NOTES:
1. Run Phase 0 (inspection) FIRST as a separate script. Read the output carefully.
2. Then adjust the final script's Data sheet references (year_header_row, series_col, first_data_col, last_data_col, data_first_row, data_last_row) based on what you actually see.
3. Also verify the Task sheet label order for rows 42-47 to confirm MIN/MAX/MEDIAN/AVERAGE/25th/75th ordering. If the labels are in a different order, rearrange the formulas accordingly.
4. CRITICAL: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) for rows 46 and 47. This was the cause of the previous failure.
5. CRITICAL: Ensure every cell assignment actually writes a formula string (starting with '=') and that `wb.save()` is called. The previous cross-task failure was caused by formulas not being committed.
6. After saving, reopen and verify that sample cells contain formula strings, not None.
7. Do NOT add any new sheets, macros, or VBA. Do NOT change formatting.

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