# Task Instruction

Execute the following steps in order.

## 0 – Inspect the workbook
```python
import openpyxl, os, shutil

wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
print('Sheet names:', wb.sheetnames)

task = wb['Task']
data = wb['Data']

# Print Task sheet layout: rows 10-50, columns D-L
print('\n--- Task sheet rows 10-50, cols D(4)-L(12) ---')
for row in task.iter_rows(min_row=10, max_row=50, min_col=4, max_col=12, values_only=False):
    vals = [(c.coordinate, c.value) for c in row]
    print(vals)

# Print Data sheet rows 19-40, cols A-Z (to find layout)
print('\n--- Data sheet rows 19-40, cols A(1)-Z(26) ---')
for row in data.iter_rows(min_row=19, max_row=40, min_col=1, max_col=26, values_only=False):
    vals = [(c.coordinate, c.value) for c in row]
    print(vals)

# Also check Data sheet row 1-5 for headers
print('\n--- Data sheet rows 1-5 ---')
for row in data.iter_rows(min_row=1, max_row=5, min_col=1, max_col=26, values_only=False):
    vals = [(c.coordinate, c.value) for c in row]
    print(vals)

wb.close()
```
Run this and paste the full output.

## 1 – Identify exact structure
After inspecting, note:
- The series codes in column D of the Task sheet (rows 12-17, 19-24, 26-31).
- The years in row 10 (columns H-L).
- The Data sheet layout: which row has headers/years, which column has series codes, and where the values are (rows 21-38).
- The labels in rows 35-40 (hospitals for net patient flow), rows 42-47 (stat labels: min, max, median, mean, 25th pctl, 75th pctl), and row 50 (weighted mean label).

## 2 – Write formulas
Using the inspection results, write a Python script that:

### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these blocks, write an INDEX/MATCH formula:
```
=INDEX(Data!$B$21:$Z$38, MATCH(D{row}, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
Adjust the ranges based on the actual Data sheet layout discovered in Step 0. The key references:
- `D{row}` = the series code in column D of the current Task row
- `H$10` (or I$10, J$10, etc.) = the year from row 10
- The Data sheet lookup array must cover rows 21:38 for values
- The row-header range (series codes) must be in column A (or wherever they are) rows 21:38
- The column-header range (years) must be in the row just above the data (row 20 or wherever years are)

Use absolute references with `$` signs appropriately. The column letter for the year reference should change per column (H, I, J, K, L) but the row should be fixed ($10). The row reference for the series code should change per row but the column should be fixed ($D).

### Step 2: Net patient flow in H35:L40
The formula for each cell is:
```
=(H{admissions_row} - H{discharges_row}) / H{capacity_row} * 100
```
where:
- admissions_row = corresponding row in H12:L17 block
- discharges_row = corresponding row in H19:L24 block  
- capacity_row = corresponding row in H26:L31 block

So for row 35: `=(H12-H19)/H26*100`, row 36: `=(H13-H20)/H27*100`, etc.

### Step 2 continued: Statistics in H42:L47
Read the labels in column D (or nearby) for rows 42-47 to confirm the order of statistics. Then for each column (H through L):
- Minimum: `=MIN(H35:H40)`
- Maximum: `=MAX(H35:H40)`
- Median: `=MEDIAN(H35:H40)`
- Mean: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)`

Map these to the correct rows based on the labels found.

### Step 3: Weighted mean in H50:L50
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
for each column H through L.

## 3 – Save
```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
wb.close()
```

## 4 – Verify
Reload the saved file and print cells H12, H19, H26, H35, H42, H50 to confirm they contain formula strings (not None).
```python
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
task2 = wb2['Task']
for r in [12,13,17,19,24,26,31,35,40,42,47,50]:
    print(f'H{r}:', task2[f'H{r}'].value)
    print(f'L{r}:', task2[f'L{r}'].value)
wb2.close()
```

## IMPORTANT NOTES
- Open the workbook ONCE for writing, write ALL formulas, then save ONCE.
- Do NOT use `data_only=True` when loading.
- All formula strings must start with `=`.
- Use `openpyxl` only (no xlsxwriter, no pandas ExcelWriter).
- Do NOT add sheets, macros, or VBA.
- Preserve all existing formatting by not touching cells outside the target ranges.
- Adjust all range references based on the actual structure found in Step 0. Do not assume the ranges I listed are exact — they are templates based on similar tasks.

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