# Task Instruction

Execute the following steps carefully to complete the task.

## 1. Inspect the workbook structure

```python
import openpyxl
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
print('Sheet names:', wb.sheetnames)

task = wb['Task']
data = wb['Data']

# Inspect Task sheet layout
print('\n--- Task sheet: Row 10 (headers/years) ---')
for col in range(1, 15):
    print(f'  {openpyxl.utils.get_column_letter(col)}10 = {task.cell(row=10, column=col).value}')

print('\n--- Task sheet: Column D (series codes) rows 12-31 ---')
for row in range(12, 32):
    print(f'  D{row} = {task.cell(row=row, column=4).value}')

print('\n--- Task sheet: Column D rows 35-50 ---')
for row in range(35, 51):
    print(f'  D{row} = {task.cell(row=row, column=4).value}')

print('\n--- Task sheet: Rows 42-47 column A-G ---')
for row in range(42, 48):
    vals = [task.cell(row=row, column=c).value for c in range(1, 8)]
    print(f'  Row {row}: {vals}')

print('\n--- Task sheet: Row 50 ---')
vals = [task.cell(row=50, column=c).value for c in range(1, 15)]
print(f'  Row 50: {vals}')

# Inspect Data sheet rows 21-38
print('\n--- Data sheet: Row 20 (header row?) ---')
for col in range(1, 25):
    v = data.cell(row=20, column=col).value
    if v is not None:
        print(f'  {openpyxl.utils.get_column_letter(col)}20 = {v}')

print('\n--- Data sheet: Rows 21-38, first 15 cols ---')
for row in range(21, 39):
    vals = [data.cell(row=row, column=c).value for c in range(1, 16)]
    print(f'  Row {row}: {vals}')

wb.close()
```

Read the output carefully. Identify:
- The exact years in row 10 of Task sheet (columns H-L).
- The series codes in column D for rows 12-17, 19-24, 26-31.
- The layout of the Data sheet rows 21-38: which column holds the series code, which columns hold years/values.
- The labels in rows 42-47 (min, max, median, mean, 25th, 75th percentile).
- The label in row 50.
- The service names in rows 35-40 and whether column D has codes or names.

## 2. Build and write the formulas

After inspecting, write a Python script that:

### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write an INDEX/MATCH formula. The formula pattern should be:
```
=INDEX(Data!$<value_col_start>:$<value_col_end>, MATCH($D<row>, Data!$<code_col>$21:$<code_col>$38, 0), MATCH(H$10, Data!$<year_row_range>, 0))
```

Adapt the exact column letters based on what you discover in the Data sheet. The key is:
- Row match: match the series code from column D of the current Task row against the series code column in Data rows 21-38.
- Column match: match the year from row 10 of the current Task column against the year header row in Data sheet.
- Return the intersecting value.

IMPORTANT: Use `$D<row>` (absolute column, relative row) and `<col>$10` (relative column, absolute row) so formulas can be placed across the grid.

Make sure you use `data_only=False` when loading and that you write formula strings (starting with `=`) to cells.

### Step 2: Net SLA buffer in H35:L40

The formula is: `(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`

Identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to each metric by checking the block labels (likely in rows 11, 18, 25 or nearby). Then for each cell in H35:L40:
```
=(<Preserved_cell> - <Consumed_cell>) / <Capacity_cell> * 100
```
where the cells reference the corresponding row in each block and the same column.

### Step 2 continued: Statistics in H42:L47

For each column (H through L), write:
- MIN of H35:H40 (or the corresponding column)
- MAX
- MEDIAN
- AVERAGE
- PERCENTILE (or PERCENTILE.INC) with 0.25
- PERCENTILE (or PERCENTILE.INC) with 0.75

Match the exact order of rows 42-47 to the labels you found.

### Step 3: Weighted mean in H50:L50

For each column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
(Adjust column letter for each column H-L.)

## 3. Save the workbook

```python
import os
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
wb.close()
```

## 4. Validate the output

Reload the saved file with `data_only=False` and verify:
- Cells H12, L17, H19, L24, H26, L31 contain formula strings (starting with `=`).
- Cells H35, L40 contain formula strings.
- Cells H42, L47 contain formula strings.
- Cell H50 contains a formula string.
- No cells in those ranges are None.

```python
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
task2 = wb2['Task']
for r in range(12, 18):
    for c in range(8, 13):
        v = task2.cell(row=r, column=c).value
        assert v is not None and str(v).startswith('='), f'Cell {openpyxl.utils.get_column_letter(c)}{r} = {v}'
for r in range(19, 25):
    for c in range(8, 13):
        v = task2.cell(row=r, column=c).value
        assert v is not None and str(v).startswith('='), f'Cell {openpyxl.utils.get_column_letter(c)}{r} = {v}'
for r in range(26, 32):
    for c in range(8, 13):
        v = task2.cell(row=r, column=c).value
        assert v is not None and str(v).startswith('='), f'Cell {openpyxl.utils.get_column_letter(c)}{r} = {v}'
for r in range(35, 41):
    for c in range(8, 13):
        v = task2.cell(row=r, column=c).value
        assert v is not None and str(v).startswith('='), f'Cell {openpyxl.utils.get_column_letter(c)}{r} = {v}'
for r in range(42, 48):
    for c in range(8, 13):
        v = task2.cell(row=r, column=c).value
        assert v is not None and str(v).startswith('='), f'Cell {openpyxl.utils.get_column_letter(c)}{r} = {v}'
for c in range(8, 13):
    v = task2.cell(row=50, column=c).value
    assert v is not None and str(v).startswith('='), f'Cell {openpyxl.utils.get_column_letter(c)}50 = {v}'
print('All validations passed!')
wb2.close()
```

## Critical Reminders

- Load the workbook with `openpyxl.load_workbook('/root/data/workbook.xlsx')` (NOT with `data_only=True`).
- Write formula strings (Python strings starting with `=`) to cells, not computed values.
- Call `wb.save(...)` before closing.
- The cross-task feedback warns that cells ended up as None — this happens if you forget to save, write to wrong cells, or use wrong coordinates. Double-check every range.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting.

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