# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Preparation

```bash
mkdir -p /root/output
pip install openpyxl
```

Then open a Python session and inspect the workbook:

```python
import openpyxl
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
print('Sheet names:', wb.sheetnames)

# Inspect Task sheet structure
ts = wb['Task']
print('--- Task sheet, rows 9-52, cols A-M ---')
for row in ts.iter_rows(min_row=9, max_row=52, min_col=1, max_col=13, values_only=False):
    for cell in row:
        print(f'{cell.coordinate}: {cell.value}', end='  |  ')
    print()

# Inspect Data sheet structure
ds = wb['Data']
print('--- Data sheet, rows 1-5 and 18-40, cols A-Z ---')
for row in ds.iter_rows(min_row=1, max_row=5, min_col=1, max_col=26, values_only=False):
    for cell in row:
        if cell.value is not None:
            print(f'{cell.coordinate}: {cell.value}', end='  |  ')
    print()
for row in ds.iter_rows(min_row=18, max_row=40, min_col=1, max_col=26, values_only=False):
    for cell in row:
        if cell.value is not None:
            print(f'{cell.coordinate}: {cell.value}', end='  |  ')
    print()
```

Read the output carefully. Identify:
- The layout of the Data sheet rows 21-38: which column holds the series code, which row holds years, and where the numeric data begins.
- On the Task sheet: what series codes are in column D for rows 12-17, 19-24, 26-31; what years are in H10:L10.
- The structure of rows 35-40 (Net renewable balance), 42-47 (stats), and 50 (weighted mean).

## 1. Write formulas into the workbook

Using openpyxl, write formulas as strings into cells. Use `INDEX(MATCH,MATCH)` pattern since it's the most robust.

**IMPORTANT**: openpyxl writes formulas as strings. When referencing another sheet, use the syntax `Data!A1`. If the sheet name has spaces, use `'Sheet Name'!A1`. Absolute references with `$` where needed for anchoring.

Based on your inspection, construct the formulas. The general pattern for each cell in H12:L17, H19:L24, H26:L31 should be:

```
=INDEX(Data!$<data_range>, MATCH($D<row>, Data!$<series_code_column>, 0), MATCH(H$10, Data!$<year_row>, 0))
```

Replace `<data_range>`, `<series_code_column>`, and `<year_row>` with the actual ranges you identified from inspection.

For example, if Data sheet has:
- Series codes in column A, rows 21-38
- Years in row 20, starting from column B
- Data values in B21:?38

Then the formula for cell H12 would be:
```
=INDEX(Data!$B$21:$<lastcol>$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$<lastcol>$20, 0))
```

Adapt based on actual inspection.

### Step 1 code pattern:
```python
import openpyxl
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
ts = wb['Task']

# After inspection, define the actual ranges. Example (ADJUST BASED ON INSPECTION):
# data_range = "Data!$B$21:$Z$38"
# code_range = "Data!$A$21:$A$38"
# year_range = "Data!$B$20:$Z$20"

# Fill H12:L17, H19:L24, H26:L31
for block_start in [12, 19, 26]:
    for r in range(block_start, block_start + 6):
        for c_idx, col_letter in enumerate(['H','I','J','K','L']):
            formula = f'=INDEX({data_range},MATCH($D{r},{code_range},0),MATCH({col_letter}$10,{year_range},0))'
            ts[f'{col_letter}{r}'] = formula
```

### Step 2: Net renewable balance in H35:L40

Based on the formula: `(Renewable Generation - Grid Consumption) / Baseline Energy Demand * 100`

Identify which row blocks correspond to:
- Renewable Generation: rows 12-17
- Grid Consumption: rows 19-24  
- Baseline Energy Demand: rows 26-31

(Verify by checking labels in column B or C of the Task sheet during inspection. The mapping might differ — adjust accordingly.)

```python
# For each campus (6 rows) and each year column (H-L):
for i in range(6):
    for col in ['H','I','J','K','L']:
        rg_row = 12 + i   # Renewable Generation (VERIFY)
        gc_row = 19 + i   # Grid Consumption (VERIFY)
        be_row = 26 + i   # Baseline Energy Demand (VERIFY)
        target_row = 35 + i
        formula = f'=({col}{rg_row}-{col}{gc_row})/{col}{be_row}*100'
        ts[f'{col}{target_row}'] = formula
```

### Step 2 continued: Statistics in H42:L47

Rows 42-47 should contain MIN, MAX, MEDIAN, AVERAGE, PERCENTILE (25th), PERCENTILE (75th) of H35:H40 through L35:L40.

```python
stat_formulas = [
    'MIN({col}35:{col}40)',
    'MAX({col}35:{col}40)',
    'MEDIAN({col}35:{col}40)',
    'AVERAGE({col}35:{col}40)',
    'PERCENTILE({col}35:{col}40,0.25)',
    'PERCENTILE({col}35:{col}40,0.75)',
]

for i, tmpl in enumerate(stat_formulas):
    for col in ['H','I','J','K','L']:
        formula = '=' + tmpl.format(col=col)
        ts[f'{col}{42+i}'] = formula
```

**IMPORTANT**: Verify the order (min, max, median, mean, 25th, 75th) matches labels in column B/C/D of rows 42-47. Adjust the order if labels differ.

### Step 3: Weighted mean in H50:L50

```python
for col in ['H','I','J','K','L']:
    formula = f'=SUMPRODUCT({col}35:{col}40,{col}26:{col}31)/SUM({col}26:{col}31)'
    ts[f'{col}50'] = formula
```

## 2. Save

```python
wb.save('/root/output/result.xlsx')
print('Saved successfully.')
```

## 3. Validate

Reopen the saved file and verify:
```python
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
ts2 = wb2['Task']

# Check formulas are present
for check_cell in ['H12','L17','H19','L24','H26','L31','H35','L40','H42','L47','H50','L50']:
    print(f'{check_cell}: {ts2[check_cell].value}')

# Verify sheet count unchanged
print('Sheets:', wb2.sheetnames)
```

Confirm:
- All formula cells contain strings starting with '='
- INDEX/MATCH formulas reference the Data sheet correctly
- Net balance formulas reference correct row blocks
- Statistics formulas are in correct order matching row labels
- SUMPRODUCT formula is in row 50
- No extra sheets were added
- File saved at /root/output/result.xlsx

**CRITICAL NOTES**:
1. You MUST inspect the workbook first before writing any formulas. The exact ranges depend on the actual layout.
2. Verify which block (rows 12-17, 19-24, 26-31) corresponds to which metric (Renewable Generation, Grid Consumption, Baseline Energy Demand) by reading labels.
3. Verify the order of statistics rows 42-47 by reading their labels.
4. Do NOT use `data_only=True` when loading — you need to preserve and write formulas.
5. Do NOT add any new sheets or modify formatting.

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