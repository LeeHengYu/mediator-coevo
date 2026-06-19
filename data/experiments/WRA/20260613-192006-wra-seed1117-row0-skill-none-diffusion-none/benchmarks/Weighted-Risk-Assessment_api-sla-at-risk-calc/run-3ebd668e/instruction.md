# Task Instruction

Execute the following Python script to inspect the workbook, build all formulas, and save the result.

```python
import openpyxl, os, shutil

# --- Phase 0: Copy workbook and inspect layout ---
os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')

wb = openpyxl.load_workbook('/root/output/result.xlsx')

# Inspect Data sheet to find the lookup range
data = wb['Data']
print('=== Data sheet rows 19-40, cols A-Z ===')
for row in data.iter_rows(min_row=19, max_row=40, min_col=1, max_col=26, values_only=False):
    vals = [(c.coordinate, c.value) for c in row if c.value is not None]
    if vals:
        print(vals)

# Inspect Task sheet structure
task = wb['Task']
print('\n=== Task sheet rows 1-55, cols A-L ===')
for row in task.iter_rows(min_row=1, max_row=55, min_col=1, max_col=12, values_only=False):
    vals = [(c.coordinate, c.value) for c in row if c.value is not None]
    if vals:
        print(vals)

wb.close()
```

After inspecting the output, run the following script (adjust Data-sheet row/column references if the inspection reveals different coordinates):

```python
import openpyxl, os
from openpyxl.utils import get_column_letter

wb = openpyxl.load_workbook('/root/output/result.xlsx')
task = wb['Task']
data = wb['Data']

# ----------------------------------------------------------------
# Phase 0-b: Confirm key coordinates from inspection
# Expected: Task!D12:D17 = series codes for block 1 (Latency Budget Preserved)
#           Task!D19:D24 = series codes for block 2 (Latency Budget Consumed)
#           Task!D26:D31 = series codes for block 3 (Covered Request Capacity)
#           Task!H10:L10 = year headers
#           Data!rows 21:38 = source records
# Print them to confirm:
print('Year headers row 10:', [task.cell(row=10, column=c).value for c in range(8,13)])
print('Block1 series D12:D17:', [task.cell(row=r, column=4).value for r in range(12,18)])
print('Block2 series D19:D24:', [task.cell(row=r, column=4).value for r in range(19,25)])
print('Block3 series D26:D31:', [task.cell(row=r, column=4).value for r in range(26,32)])

# Identify Data sheet layout: find the column with series codes and the row with years
# Print Data row 21 header and first few data rows
print('\nData row 20 (likely header):', [(get_column_letter(c), data.cell(row=20, column=c).value) for c in range(1,20)])
print('Data row 21:', [(get_column_letter(c), data.cell(row=21, column=c).value) for c in range(1,20)])
print('Data row 22:', [(get_column_letter(c), data.cell(row=22, column=c).value) for c in range(1,20)])
print('Data row 38:', [(get_column_letter(c), data.cell(row=38, column=c).value) for c in range(1,20)])

wb.close()
```

After confirming the layout, run the final formula-writing script. The template below assumes the standard layout (adjust if needed based on inspection):

```python
import openpyxl, os
from openpyxl.utils import get_column_letter

wb = openpyxl.load_workbook('/root/output/result.xlsx')
task = wb['Task']

# ----------------------------------------------------------------
# Step 1: Populate H12:L17, H19:L24, H26:L31 with INDEX/MATCH formulas
# Pattern: =INDEX(Data!<data_range>, MATCH(D{row}, Data!<series_col>, 0), MATCH(H$10, Data!<year_row>, 0))
# 
# We need to identify from inspection:
#   - The column in Data that holds series codes (call it SC, e.g. column A = $A$21:$A$38)
#   - The row in Data that holds year headers (call it YR, e.g. row 20 = $B$20:$??$20)
#   - The data block range (e.g. $B$21:$??$38)
# These will be filled in after Phase 0 inspection.
# 
# PLACEHOLDER references (update after inspection):
#   data_series_col = "Data!$A$21:$A$38"  (series codes)
#   data_year_row   = "Data!$B$20:$??$20" (year headers)
#   data_block      = "Data!$B$21:$??$38" (values)

# --- FILL THESE FROM INSPECTION ---
# Example: if Data has series codes in col A rows 21-38, years in row 20 cols B-F,
# and data values in B21:F38:
DATA_SERIES = 'Data!$A$21:$A$38'   # adjust column letter and rows
DATA_YEARS  = 'Data!$B$20:$F$20'   # adjust column letters
DATA_BLOCK  = 'Data!$B$21:$F$38'   # adjust column letters

# Blocks to fill: (start_row, end_row)
blocks = [(12, 17), (19, 24), (26, 31)]
col_start = 8  # H
col_end = 12    # L

for (r_start, r_end) in blocks:
    for r in range(r_start, r_end + 1):
        for c in range(col_start, col_end + 1):
            col_letter = get_column_letter(c)
            # D{r} has the series code; {col_letter}$10 has the year
            formula = f'=INDEX({DATA_BLOCK},MATCH(D{r},{DATA_SERIES},0),MATCH({col_letter}$10,{DATA_YEARS},0))'
            task.cell(row=r, column=c).value = formula

# ----------------------------------------------------------------
# Step 2: H35:L40 = Net SLA buffer
# Formula: (Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100
# Block 1 rows 12-17, Block 2 rows 19-24, Block 3 rows 26-31
# Service i (0-5): preserved = row 12+i, consumed = row 19+i, capacity = row 26+i
# Net SLA buffer row = 35+i

for i in range(6):
    pres_row = 12 + i
    cons_row = 19 + i
    cap_row  = 26 + i
    out_row  = 35 + i
    for c in range(col_start, col_end + 1):
        cl = get_column_letter(c)
        formula = f'=({cl}{pres_row}-{cl}{cons_row})/{cl}{cap_row}*100'
        task.cell(row=out_row, column=c).value = formula

# ----------------------------------------------------------------
# Step 3: H42:L47 = min, max, median, mean, 25th pctl, 75th pctl (column-wise)
# Check the labels in column D or nearby to confirm the order.
# Common order: min, max, median, mean, 25th percentile, 75th percentile
# Rows 42-47

# Print labels to confirm order
print('Stat labels:')
for r in range(42, 48):
    print(f'  Row {r}:', task.cell(row=r, column=4).value, task.cell(row=r, column=3).value, task.cell(row=r, column=2).value)

# We'll assign based on typical order; adjust if labels differ
# Using plain PERCENTILE (not .INC/.EXC) to avoid #NAME? errors
stat_funcs = {
    42: 'MIN',
    43: 'MAX',
    44: 'MEDIAN',
    45: 'AVERAGE',
    46: 'PERCENTILE',  # 25th
    47: 'PERCENTILE',  # 75th
}

for c in range(col_start, col_end + 1):
    cl = get_column_letter(c)
    rng = f'{cl}35:{cl}40'
    task.cell(row=42, column=c).value = f'=MIN({rng})'
    task.cell(row=43, column=c).value = f'=MAX({rng})'
    task.cell(row=44, column=c).value = f'=MEDIAN({rng})'
    task.cell(row=45, column=c).value = f'=AVERAGE({rng})'
    task.cell(row=46, column=c).value = f'=PERCENTILE({rng},0.25)'
    task.cell(row=47, column=c).value = f'=PERCENTILE({rng},0.75)'

# ----------------------------------------------------------------
# Step 4: H50:L50 = Weighted mean using SUMPRODUCT
# Values = Net SLA buffer (H35:L40 for each column)
# Weights = Covered Request Capacity (H26:L31 for each column)
# =SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)

for c in range(col_start, col_end + 1):
    cl = get_column_letter(c)
    vals = f'{cl}35:{cl}40'
    wts  = f'{cl}26:{cl}31'
    formula = f'=SUMPRODUCT({vals},{wts})/SUM({wts})'
    task.cell(row=50, column=c).value = formula

# ----------------------------------------------------------------
# Save
wb.save('/root/output/result.xlsx')
wb.close()
print('\nDone. Saved to /root/output/result.xlsx')
```

IMPORTANT EXECUTION NOTES:
1. Run Phase 0 inspection FIRST. Read the output carefully.
2. Before running the final formula script, adjust these references based on inspection:
   - `DATA_SERIES`: The column range in Data sheet containing series/code identifiers (rows 21-38)
   - `DATA_YEARS`: The row in Data sheet containing year headers
   - `DATA_BLOCK`: The rectangular data range in Data sheet
3. Also verify the stat labels in rows 42-47 match the assumed order (min, max, median, mean, 25th pctl, 75th pctl). If different, reorder accordingly.
4. Use plain `PERCENTILE` (NOT `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid #NAME? errors.
5. After saving, do a quick verification read:

```python
import openpyxl
wb = openpyxl.load_workbook('/root/output/result.xlsx')
task = wb['Task']
print('Verification samples:')
for r in [12, 19, 26, 35, 42, 46, 50]:
    for c in [8, 12]:
        print(f'  {task.cell(row=r, column=c).coordinate}: {task.cell(row=r, column=c).value}')
wb.close()
```

Confirm all sampled cells contain formula strings (starting with '='), not None.

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