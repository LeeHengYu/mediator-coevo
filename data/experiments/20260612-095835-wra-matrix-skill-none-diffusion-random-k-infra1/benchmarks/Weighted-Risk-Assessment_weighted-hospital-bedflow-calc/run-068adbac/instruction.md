# Task Instruction

Execute the following steps exactly, in order.

## 0. Inspect the workbook
```bash
pip install openpyxl
```
```python
import openpyxl, pprint
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
print('Sheet names:', wb.sheetnames)
ts = wb[wb.sheetnames[0]]  # likely 'Task'
ds = wb[wb.sheetnames[1]]  # likely 'Data'
print('--- Task sheet name:', ts.title)
print('--- Data sheet name:', ds.title)

# Print row 10 headers (years) in columns H-L
for col in ['H','I','J','K','L']:
    print(f'Task {col}10 =', ts[f'{col}10'].value)

# Print column D series codes for rows 12-17, 19-24, 26-31
for r in list(range(12,18))+list(range(19,25))+list(range(26,32)):
    print(f'Task D{r} =', ts[f'D{r}'].value)

# Print Data sheet row 21-38 to understand layout
for r in range(20, 39):
    row_vals = [ds.cell(row=r, column=c).value for c in range(1, 15)]
    print(f'Data row {r}: {row_vals}')

# Print Task rows 35-50 column D (labels)
for r in range(35, 51):
    print(f'Task D{r} =', ts[f'D{r}'].value)

# Print rows 42-47 col C-D for stat labels
for r in range(42, 48):
    print(f'Task C{r}={ts[f"C{r}"].value}  D{r}={ts[f"D{r}"].value}')

# Print row 50
for c in ['C','D','E','F','G']:
    print(f'Task {c}50 =', ts[f'{c}50'].value)

# Check H26:L31 for existing content
for r in range(26,32):
    for col in ['H','I','J','K','L']:
        print(f'Task {col}{r} =', ts[f'{col}{r}'].value)

wb.close()
```

## 1. Write formulas into the workbook

After inspecting, write a Python script that:

1. Opens `/root/data/workbook.xlsx` with `openpyxl.load_workbook('/root/data/workbook.xlsx')`.
2. Gets the Task sheet **by its exact title** (from step 0) and the Data sheet **by its exact title**.
3. Determines the exact Data sheet name string to use inside formulas (e.g., `Data` → use `Data` in formula references like `Data!A21:A38`).

### Step 1 — Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell at row `r`, column `c` (H=8, I=9, J=10, K=11, L=12):
- The series code is in `$D{r}` on the Task sheet.
- The year is in `{col_letter}$10` on the Task sheet.
- The data lives in Data sheet rows 21:38.

Use this formula pattern (adjust the Data sheet name if needed):
```
=INDEX(Data!$A$21:$Z$38, MATCH($D{r}, Data!$A$21:$A$38, 0), MATCH({col}$10, Data!$A$20:$Z$20, 0))
```
**Important**: Before writing this formula, verify from the inspection output:
- Which column in the Data sheet holds the series codes (likely column A or B). Adjust the MATCH range accordingly.
- Which row in the Data sheet holds the year headers (likely row 20 or 21). Adjust accordingly.
- How wide the data extends (adjust $Z to the actual last column).

Write the formula as a string into each cell. Make sure the Data sheet name in the formula matches exactly (case-sensitive). If the sheet name contains spaces, wrap it in single quotes like `'Sheet Name'!`.

### Step 2 — Net patient flow in H35:L40

For each hospital (rows 35-40), the net patient flow formula references the corresponding rows from the three lookup blocks:
- Patient Admissions = rows 12-17 (H12:L17)
- Patient Discharges = rows 19-24 (H19:L24)  
- Effective Bed Capacity = rows 26-31 (H26:L31)

So for row 35 (first hospital), column H:
```
=(H12-H19)/H26*100
```
For row 36: `=(H13-H20)/H27*100`, etc.

Write these for all cells H35:L40.

### Step 2 continued — Statistics in H42:L47

For each column (H through L):
- Row 42 (Min): `=MIN(H35:H40)`
- Row 43 (Max): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**Verify from the inspection** which row maps to which statistic (min/max/median/mean/25th/75th). Use the labels found in column C or D of rows 42-47 to assign the correct function.

### Step 3 — Weighted mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## 2. Save

```python
import os
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
wb.close()
```

## 3. Verify

Reopen the saved file and check that the formula cells are not None:
```python
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
ts2 = wb2[wb2.sheetnames[0]]
for cell in ['H12','L17','H19','L24','H26','L31','H35','L40','H42','L47','H50','L50']:
    val = ts2[cell].value
    print(f'{cell} = {val}')
    assert val is not None, f'{cell} is None!'
wb2.close()
print('All checks passed.')
```

If any cell is None, debug by printing the cell and re-examining the formula writing code. Common issues:
- Wrong sheet name reference in the formula string
- Writing to wrong sheet object
- Not saving before closing

## 4. Run the verifier if available
```bash
cd /root && python -m pytest test_output.py -v 2>&1 | head -80
```
If tests fail, read the error messages carefully, fix the formulas, and re-save.

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