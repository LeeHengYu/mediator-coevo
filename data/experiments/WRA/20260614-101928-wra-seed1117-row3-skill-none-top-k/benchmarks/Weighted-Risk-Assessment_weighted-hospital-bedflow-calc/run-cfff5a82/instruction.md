# Task Instruction

Execute the following Python script in a single block to inspect the Data sheet layout, build correct formulas, and save the result.

```python
import openpyxl
import shutil
import os

# --- Step 0: Copy workbook to output ---
os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')

# --- Step 1: Inspect the Data sheet to understand layout ---
wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws_data = wb['Data']
ws_task = wb['Task']

# Print Data sheet rows 1-40 to understand structure
print('=== DATA SHEET INSPECTION ===')
for row in range(1, 45):
    vals = []
    for col in range(1, 20):
        cell = ws_data.cell(row=row, column=col)
        vals.append(f'{cell.value}')
    print(f'Row {row}: {vals}')

print('\n=== TASK SHEET INSPECTION ===')
# Print Task sheet rows 1-55
for row in range(1, 56):
    vals = []
    for col in range(1, 15):
        cell = ws_task.cell(row=row, column=col)
        vals.append(f'{cell.value}')
    print(f'Row {row}: {vals}')

wb.close()
```

After inspecting the output, execute this second script that writes all formulas and saves:

```python
import openpyxl
from openpyxl.utils import get_column_letter

wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws_task = wb['Task']

# --- Gather info from inspection ---
# Task sheet layout (expected based on benchmark description):
#   Row 10: header row with years in columns H(8) through L(12)
#   Column D: series codes
#   H12:L17 = first lookup block (6 rows)
#   H19:L24 = second lookup block (6 rows)
#   H26:L31 = third lookup block (6 rows)
#   Data sheet rows 21:38 contain the source data

# We need to identify the Data sheet structure:
# - Which column has the series codes?
# - Which row has the year headers?
# - Where is the actual data?

# Let's read the Data sheet to find the series code column and year header row
ws_data = wb['Data']

# Find the column in Data sheet that contains series codes
# and the row that contains year headers
# We'll look at rows 21-38 as specified

# Print specific cells for debugging
print('Task D12:', ws_task.cell(row=12, column=4).value)
print('Task D13:', ws_task.cell(row=13, column=4).value)
print('Task H10:', ws_task.cell(row=10, column=8).value)
print('Task I10:', ws_task.cell(row=10, column=9).value)
print('Task J10:', ws_task.cell(row=10, column=10).value)
print('Task K10:', ws_task.cell(row=10, column=11).value)
print('Task L10:', ws_task.cell(row=10, column=12).value)

# Check Data sheet structure around rows 21-38
for r in range(19, 40):
    row_vals = []
    for c in range(1, 15):
        row_vals.append((get_column_letter(c), ws_data.cell(row=r, column=c).value))
    print(f'Data row {r}: {row_vals}')

# Also check the header area of Data sheet
for r in range(1, 22):
    row_vals = []
    for c in range(1, 15):
        row_vals.append((get_column_letter(c), ws_data.cell(row=r, column=c).value))
    print(f'Data row {r}: {row_vals}')

wb.close()
```

After reviewing both inspection outputs, execute the final script below. You MUST adapt the column letters and row numbers based on what you actually see in the inspection. The template below uses placeholders that you need to replace:

```python
import openpyxl
from openpyxl.utils import get_column_letter

wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws_task = wb['Task']

# ============================================================
# IMPORTANT: Replace these variables based on inspection output
# ============================================================
# SERIES_COL_DATA = the column letter in Data sheet where series codes are
# YEAR_ROW_DATA = the row number in Data sheet where year headers are  
# DATA_START_COL = first data column letter in Data sheet
# DATA_END_COL = last data column letter in Data sheet
# DATA_ROW_START = 21 (as specified)
# DATA_ROW_END = 38 (as specified)
# ============================================================

# After inspection, set these correctly. Example assumptions:
# (You MUST verify and adjust these from the inspection output)

# Read actual values from Task sheet
year_cols = {8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L'}  # columns H-L

# Step 1: Write INDEX/MATCH formulas for the three lookup blocks
# Blocks: rows 12-17, 19-24, 26-31 on Task sheet
lookup_rows = list(range(12, 18)) + list(range(19, 25)) + list(range(26, 32))

for row in lookup_rows:
    for col_num in range(8, 13):  # H=8 through L=12
        col_letter = get_column_letter(col_num)
        # Formula: INDEX(MATCH()) pattern
        # Look up series code from column D of current row
        # Look up year from row 10 of current column
        # Data is in Data!$21:$38
        # We need to know which column in Data has series codes and which row has years
        # Using INDEX/MATCH/MATCH for a two-dimensional lookup:
        # =INDEX(Data!<data_range>, MATCH($D<row>, Data!<series_col>, 0), MATCH(<col>$10, Data!<year_row>, 0))
        
        # PLACEHOLDER - replace SERIES_COL, YEAR_ROW, and data range after inspection
        # This is the pattern:
        # =INDEX(Data!$B$21:$N$38, MATCH($D{row}, Data!$A$21:$A$38, 0), MATCH({col_letter}$10, Data!$B$20:$N$20, 0))
        
        formula = f'=INDEX(Data!$B$21:$N$38,MATCH($D{row},Data!$A$21:$A$38,0),MATCH({col_letter}$10,Data!$B$20:$N$20,0))'
        ws_task.cell(row=row, column=col_num).value = formula
        print(f'Wrote {col_letter}{row}: {formula}')

# Step 2: Net patient flow = (Admissions - Discharges) / Effective Bed Capacity * 100
# Admissions block: rows 12-17, Discharges block: rows 19-24, Bed Capacity block: rows 26-31
# Net flow goes in rows 35-40
for i in range(6):
    admit_row = 12 + i
    discharge_row = 19 + i
    capacity_row = 26 + i
    flow_row = 35 + i
    for col_num in range(8, 13):
        col_letter = get_column_letter(col_num)
        formula = f'=({col_letter}{admit_row}-{col_letter}{discharge_row})/{col_letter}{capacity_row}*100'
        ws_task.cell(row=flow_row, column=col_num).value = formula
        print(f'Wrote {col_letter}{flow_row}: {formula}')

# Step 2 continued: Statistics in rows 42-47
# Row 42: MIN, Row 43: MAX, Row 44: MEDIAN, Row 45: AVERAGE, Row 46: 25th percentile, Row 47: 75th percentile
# But verify the labels from inspection! Adjust row assignments if needed.
stat_formulas = {
    42: 'MIN',
    43: 'MAX', 
    44: 'MEDIAN',
    45: 'AVERAGE',
    46: 'PERCENTILE',  # 25th
    47: 'PERCENTILE',  # 75th
}

for col_num in range(8, 13):
    col_letter = get_column_letter(col_num)
    data_range = f'{col_letter}35:{col_letter}40'
    
    ws_task.cell(row=42, column=col_num).value = f'=MIN({data_range})'
    ws_task.cell(row=43, column=col_num).value = f'=MAX({data_range})'
    ws_task.cell(row=44, column=col_num).value = f'=MEDIAN({data_range})'
    ws_task.cell(row=45, column=col_num).value = f'=AVERAGE({data_range})'
    ws_task.cell(row=46, column=col_num).value = f'=PERCENTILE({data_range},0.25)'
    ws_task.cell(row=47, column=col_num).value = f'=PERCENTILE({data_range},0.75)'
    
    print(f'Wrote stats for column {col_letter}')

# Step 3: Weighted mean in row 50 using SUMPRODUCT
# Values = net patient flow (H35:H40 etc), Weights = Effective Bed Capacity (H26:H31 etc)
for col_num in range(8, 13):
    col_letter = get_column_letter(col_num)
    flow_range = f'{col_letter}35:{col_letter}40'
    weight_range = f'{col_letter}26:{col_letter}31'
    formula = f'=SUMPRODUCT({flow_range},{weight_range})/SUM({weight_range})'
    ws_task.cell(row=50, column=col_num).value = formula
    print(f'Wrote {col_letter}50: {formula}')

wb.save('/root/output/result.xlsx')
print('\nWorkbook saved successfully!')

# Verify formulas were written
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb2['Task']
for cell_ref in ['H12', 'L17', 'H19', 'L24', 'H26', 'L31', 'H35', 'L40', 'H42', 'L47', 'H50', 'L50']:
    from openpyxl.utils import coordinate_to_tuple
    r, c = coordinate_to_tuple(cell_ref)
    val = ws.cell(row=r, column=c).value
    print(f'{cell_ref}: {val}')
wb2.close()
```

CRITICAL INSTRUCTIONS FOR THE EXECUTOR:

1. Run the FIRST inspection script and carefully read ALL output before proceeding.
2. Run the SECOND inspection script and carefully read ALL output.
3. Based on the inspection output, ADAPT the third script before running it:
   - Identify which column in the Data sheet contains the series codes (might be column A, B, or another).
   - Identify which row in the Data sheet contains the year headers (might be row 20, or another row within/near 21-38).
   - Identify the data range boundaries in the Data sheet.
   - Adjust the INDEX/MATCH formula template accordingly.
   - Verify the Task sheet labels for rows 42-47 match the stat functions (MIN, MAX, MEDIAN, AVERAGE, 25th percentile, 75th percentile) and reorder if needed.
   - Check if the lookup blocks really correspond to Admissions (12-17), Discharges (19-24), and Bed Capacity (26-31) by reading the Task sheet labels.
4. After saving, verify that cells contain formulas (strings starting with '='), not None.
5. If any cell shows None in the verification, debug and fix before finishing.
6. Do NOT use _xlfn. prefixes for PERCENTILE - use plain PERCENTILE.
7. The PERCENTILE function arguments should be (range, 0.25) and (range, 0.75).
8. Make sure the Data sheet reference range in INDEX formulas covers all 18 rows (21-38) and all necessary columns.

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