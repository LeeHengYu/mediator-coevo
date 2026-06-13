# Task Instruction

Execute the following steps to complete the API SLA at-risk calculation workbook.

## 0 — Inspect the workbook
```bash
cd /root
python3 - <<'PY'
import openpyxl, json
wb = openpyxl.load_workbook('data/workbook.xlsx')
for s in wb.sheetnames:
    print(f'=== Sheet: {s} ===')
    ws = wb[s]
    print(f'  Dimensions: {ws.dimensions}')
    # Print key header rows and data layout
    for r in range(1, min(ws.max_row+1, 55)):
        vals = []
        for c in range(1, min(ws.max_column+1, 15)):
            cell = ws.cell(r, c)
            vals.append(f'{cell.value}')
        print(f'  Row {r:3d}: {vals}')
PY
```
Study the output carefully. Identify:
- The series codes in column D for rows 12-17, 19-24, 26-31, 35-40.
- The years in row 10 for columns H-L.
- The Data sheet layout (rows 21-38): which column holds the series code, and how the year headers and values are arranged.
- The labels in rows 42-47 (min, max, median, mean, 25th percentile, 75th percentile — note the exact order).
- Row 50 label for Platform SLA Coalition weighted mean.

## 1 — Write the formulas with openpyxl
```bash
python3 - <<'PYEOF'
import openpyxl
import shutil, os

os.makedirs('/root/output', exist_ok=True)
shutil.copy2('/root/data/workbook.xlsx', '/root/output/result.xlsx')

wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb['Task']
data_sheet = wb['Data']

# --- Understand the Data sheet layout ---
# Print Data rows 21-38 first few columns to confirm structure
for r in range(19, 40):
    row_vals = [data_sheet.cell(r, c).value for c in range(1, 15)]
    print(f'Data row {r}: {row_vals}')

# Print Task row 10 (year headers) and column D series codes
print('Task row 10:', [ws.cell(10, c).value for c in range(1, 15)])
for r in range(12, 32):
    print(f'Task row {r} col D: {ws.cell(r, 4).value}')
for r in range(35, 48):
    print(f'Task row {r} col A-D: {[ws.cell(r, c).value for c in range(1, 5)]}')
print(f'Task row 50 col A-G: {[ws.cell(50, c).value for c in range(1, 8)]}')

wb.close()
PYEOF
```
Read all this output, then proceed.

## 2 — Populate formulas
Using the layout discovered above, write a Python script that:

### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For each yellow cell at row `r`, column `c` (H=8 … L=12):
- The series code is in `$D{r}` on sheet Task.
- The year is in `{col_letter}$10` on sheet Task.
- Data is on sheet Data, rows 21:38. Determine which column holds the series code and which row holds the year headers.
- Use `INDEX(Data!<value_range>, MATCH($D{r}, Data!<series_code_column>, 0), MATCH({col_letter}$10, Data!<year_header_row>, 0))` pattern.
- Make sure the Data range references are correct (absolute where needed).

### Step 2: Net SLA buffer in H35:L40
For each cell at row `r` in 35-40, column `c` in H-L:
- Identify which rows in the Task sheet correspond to "Latency Budget Preserved" (rows 12-17), "Latency Budget Consumed" (rows 19-24), and "Covered Request Capacity" (rows 26-31).
- The service index offset: row 35 corresponds to the 1st service, row 40 to the 6th.
- Formula: `=({col}{preserved_row} - {col}{consumed_row}) / {col}{capacity_row} * 100`
  where preserved_row = r - 23 (i.e., 35→12, 36→13, …, 40→17), consumed_row = r - 16 (35→19, …, 40→24), capacity_row = r - 9 (35→26, …, 40→31).
- Verify these offsets against the actual row numbers discovered.

### Step 2 continued: Statistics in H42:L47
For each column `c` (H-L), using the range `{col}35:{col}40`:
- Determine the exact order of statistics from the labels in column A/B/D of rows 42-47.
- Use these Excel functions (note: use `_xlfn.` prefix for PERCENTILE.INC if needed, but try without first; the previous failure artifact warns about #NAME? errors from incorrect function names):
  - MIN: `=MIN({col}35:{col}40)`
  - MAX: `=MAX({col}35:{col}40)`
  - MEDIAN: `=MEDIAN({col}35:{col}40)`
  - AVERAGE: `=AVERAGE({col}35:{col}40)`
  - 25th percentile: `=PERCENTILE({col}35:{col}40, 0.25)` — use `PERCENTILE` not `PERCENTILE.INC` to avoid #NAME? errors in openpyxl
  - 75th percentile: `=PERCENTILE({col}35:{col}40, 0.75)`
- Map each statistic to the correct row based on the labels discovered.

### Step 3: Weighted mean in H50:L50
For each column `c` (H-L):
`=SUMPRODUCT({col}35:{col}40, {col}26:{col}31) / SUM({col}26:{col}31)`

### Important notes:
- Use `openpyxl` to write formulas as strings (e.g., `ws['H12'] = '=INDEX(...)'`).
- Do NOT use `_xlfn.` prefix for standard functions (MIN, MAX, MEDIAN, AVERAGE, PERCENTILE) — this avoids #NAME? errors.
- Preserve all existing formatting — do not touch cell styles, fills, fonts, borders, number formats.
- Only write to the yellow cells specified.
- Save to `/root/output/result.xlsx`.

## 3 — Validate
```bash
cd /root && python3 -c "
import openpyxl
wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb['Task']
# Check a sample of cells have formulas
for r, c in [(12,8),(17,12),(19,8),(24,12),(26,8),(31,12),(35,8),(40,12),(42,8),(47,12),(50,8),(50,12)]:
    v = ws.cell(r, c).value
    print(f'  ({r},{c}): {v}')
    assert isinstance(v, str) and v.startswith('='), f'Cell ({r},{c}) is not a formula: {v}'
print('All sample cells contain formulas.')
"
```

Then run the verifier if available:
```bash
cd /root && python3 -m pytest tests/ -v 2>&1 | head -80
```

If any test fails, read the error, fix the formulas, and re-run. Pay special attention to:
- #NAME? errors (use PERCENTILE not PERCENTILE.INC)
- Wrong cell references (double-check row/column offsets)
- Data sheet range boundaries

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