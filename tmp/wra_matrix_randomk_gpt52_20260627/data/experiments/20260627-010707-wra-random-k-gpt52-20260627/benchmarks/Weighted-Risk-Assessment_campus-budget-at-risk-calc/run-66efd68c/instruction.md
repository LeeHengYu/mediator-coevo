# Task Instruction

Execute the following steps exactly, in order.

## 0 – Inspect the workbook
```bash
cd /root/data
python3 - <<'PY'
import openpyxl, pprint
wb = openpyxl.load_workbook('workbook.xlsx', data_only=False)
for s in wb.sheetnames:
    print(f'=== Sheet: {s} ===')
    ws = wb[s]
    print(f'  dims: {ws.dimensions}')
    # Print first 50 rows (columns A-M)
    for row in ws.iter_rows(min_row=1, max_row=max(ws.max_row,50), min_col=1, max_col=13, values_only=False):
        vals = [(c.coordinate, c.value) for c in row if c.value is not None]
        if vals:
            print(' ', vals)
PY
```
Read the output carefully. Identify:
- The series codes in column D for rows 12-17, 19-24, 26-31.
- The years in row 10 for columns H-L.
- The layout of the Data sheet rows 21-38 (which column holds the series code, which row holds years, how data is arranged).
- The labels in rows 35-40 (departments), row 42-47 (stats), row 50 (weighted mean).
- The three blocks: Committed Funding (rows 12-17), Operating Spend (rows 19-24), Approved Budget Base (rows 26-31).

## 1 – Build the result workbook
```bash
python3 - <<'PYEOF'
import openpyxl
import copy
import shutil
import os

# Load preserving styles
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
ws = wb['Task']
data_ws = wb['Data']

# ---- Understand Data sheet layout ----
# Print rows 1-5 and 18-40 of Data sheet for column structure
print("DATA SHEET INSPECTION:")
for r in range(1, 6):
    row_vals = {}
    for c in range(1, data_ws.max_column+1):
        v = data_ws.cell(r, c).value
        if v is not None:
            row_vals[openpyxl.utils.get_column_letter(c)] = v
    if row_vals:
        print(f"  Row {r}: {row_vals}")
for r in range(18, 40):
    row_vals = {}
    for c in range(1, min(data_ws.max_column+1, 30)):
        v = data_ws.cell(r, c).value
        if v is not None:
            row_vals[openpyxl.utils.get_column_letter(c)] = v
    if row_vals:
        print(f"  Row {r}: {row_vals}")

print("\nTASK SHEET INSPECTION:")
for r in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]:
    row_vals = {}
    for c in range(1, 14):
        v = ws.cell(r, c).value
        if v is not None:
            row_vals[openpyxl.utils.get_column_letter(c)] = v
    if row_vals:
        print(f"  Row {r}: {row_vals}")

PYEOF
```
Read the output carefully before proceeding.

## 2 – Write the formulas
Based on the inspection above, write a Python script that:

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these ranges, write an INDEX/MATCH formula that:
- Looks up the series code from column D of the same row
- Looks up the year from row 10 of the same column
- Searches in Data!$21:$38 (or whichever range is correct based on inspection)

Use this pattern (adjust column/row references based on inspection):
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```

Make sure to use absolute references for the lookup arrays and mixed references ($D12, H$10) so formulas copy correctly across the block.

### Step 2 – Net budget buffer in H35:L40
Formula: `=(H12-H19)/H26*100` (adjusted per row for the 6 departments, i.e., row 35 uses rows 12,19,26; row 36 uses rows 13,20,27; etc.)

### Step 2 – Summary stats in H42:L47
For each column H through L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Use `PERCENTILE` exactly as spelled. Do NOT use `PERCENTILE.INC` or `PERCINTILE`. The function name must be exactly `PERCENTILE`. If the evaluation engine still rejects it, also try `_xlfn.PERCENTILE.INC` as a fallback (see below).

### Step 3 – Weighted mean in H50:L50
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```

### Important implementation notes:
1. Read the actual cell coordinates from inspection before writing formulas.
2. Do NOT add sheets, macros, VBA, or external links.
3. Preserve all existing formatting.
4. Save to `/root/output/result.xlsx`.

Here is the template script (fill in after inspection):

```python
import openpyxl
import os

wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
ws = wb['Task']

# After confirming layout from inspection, set formulas.
# Adjust all references based on actual Data sheet structure.

# Example for INDEX/MATCH (adjust data_range, series_col, year_row):
# =INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))

# Step 1: Lookup blocks
for block_start in [12, 19, 26]:  # three blocks each 6 rows
    for r in range(block_start, block_start + 6):
        for c_idx, col_letter in enumerate(['H','I','J','K','L']):
            col_num = 8 + c_idx  # H=8
            # Build formula - ADJUST references based on inspection
            formula = f'=INDEX(Data!$B$21:$XX$38,MATCH($D{r},Data!$A$21:$A$38,0),MATCH({col_letter}$10,Data!$B$20:$XX$20,0))'
            ws.cell(row=r, column=col_num).value = formula

# Step 2a: Net budget buffer
for i in range(6):
    for c_idx, col_letter in enumerate(['H','I','J','K','L']):
        col_num = 8 + c_idx
        r_committed = 12 + i
        r_operating = 19 + i
        r_budget = 26 + i
        r_target = 35 + i
        formula = f'=({col_letter}{r_committed}-{col_letter}{r_operating})/{col_letter}{r_budget}*100'
        ws.cell(row=r_target, column=col_num).value = formula

# Step 2b: Summary statistics
stat_funcs = [
    'MIN',       # row 42
    'MAX',       # row 43
    'MEDIAN',    # row 44
    'AVERAGE',   # row 45
    'PERCENTILE',# row 46 - 25th percentile
    'PERCENTILE',# row 47 - 75th percentile
]
stat_rows = [42, 43, 44, 45, 46, 47]
for idx, (func, row) in enumerate(zip(stat_funcs, stat_rows)):
    for c_idx, col_letter in enumerate(['H','I','J','K','L']):
        col_num = 8 + c_idx
        rng = f'{col_letter}35:{col_letter}40'
        if func == 'PERCENTILE' and row == 46:
            formula = f'=PERCENTILE({rng},0.25)'
        elif func == 'PERCENTILE' and row == 47:
            formula = f'=PERCENTILE({rng},0.75)'
        else:
            formula = f'={func}({rng})'
        ws.cell(row=row, column=col_num).value = formula

# Step 3: Weighted mean
for c_idx, col_letter in enumerate(['H','I','J','K','L']):
    col_num = 8 + c_idx
    formula = f'=SUMPRODUCT({col_letter}35:{col_letter}40,{col_letter}26:{col_letter}31)/SUM({col_letter}26:{col_letter}31)'
    ws.cell(row=50, column=col_num).value = formula

os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
print('Saved result.xlsx')
```

## 3 – Validate
```bash
python3 - <<'PY'
import openpyxl
wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb['Task']
print("Sample formulas:")
for r in [12, 19, 26, 35, 42, 43, 44, 45, 46, 47, 50]:
    for c in [8, 9, 10, 11, 12]:
        print(f"  {openpyxl.utils.get_column_letter(c)}{r}: {ws.cell(r,c).value}")

# Verify PERCENTILE spelling
for r in [46, 47]:
    for c in [8]:
        val = str(ws.cell(r,c).value)
        assert 'PERCENTILE' in val.upper(), f"Row {r} missing PERCENTILE: {val}"
        assert 'PERCINTILE' not in val.upper(), f"Row {r} has typo: {val}"
        print(f"  Row {r} PERCENTILE check OK: {val}")
print("Validation passed.")
PY
```

## CRITICAL NOTES
- The previous run failed because PERCENTILE formulas produced #NAME? errors. Make absolutely sure the function name is spelled exactly `PERCENTILE` (not `PERCINTILE`, not `PERCENTILE.INC` without prefix).
- If after inspection you discover the evaluation engine uses `_xlfn.` prefixed functions, use `_xlfn.PERCENTILE.INC` for the percentile formulas instead of bare `PERCENTILE`.
- Adjust ALL cell references and ranges based on the actual workbook layout discovered in steps 0 and 1. Do NOT blindly use the template references.
- The Data sheet range for INDEX/MATCH must cover the actual data area. Check where series codes are and where years are.
- Ensure the output directory exists before saving.

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