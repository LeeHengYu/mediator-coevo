# Task Instruction

Execute the following steps exactly, in order.

## 0. Inspect the workbook
```python
import openpyxl, pprint
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for s in wb.sheetnames:
    print('Sheet:', s)
ws_task = wb['Task']
ws_data = wb['Data']

# Print Task sheet row 10 (years header)
print('Task row 10:', [ws_task.cell(r=10,c=c).value for c in range(1,15)])
# Print Task column D rows 12-31 (series codes)
for r in range(12,32):
    print(f'Task D{r}:', ws_task.cell(r=r,column=4).value)
# Print Task rows 35-50 col A-G for labels
for r in range(35,51):
    print(f'Task row {r}:', [ws_task.cell(r=r,c=c).value for c in range(1,8)])

# Print Data sheet structure around rows 21-38
print('\nData row 1 (header?):', [ws_data.cell(r=1,c=c).value for c in range(1,20)])
for r in range(19,40):
    print(f'Data row {r}:', [ws_data.cell(r=r,c=c).value for c in range(1,20)])

wb.close()
```
Run this and read all output carefully before proceeding.

## 1. Understand the Data sheet layout
Identify:
- Which column holds the series codes (the lookup key matching Task col D).
- Which row holds the years (matching Task row 10 years).
- The data value columns/rows in Data rows 21:38.

## 2. Write the lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in those ranges, write a formula that combines two MATCH calls with INDEX (or VLOOKUP+MATCH, etc.):
- One MATCH finds the column position of the year (from Task row 10) in the Data sheet's year header row.
- One MATCH finds the row position of the series code (from Task column D) in the Data sheet's series-code column.
- INDEX (or equivalent) returns the intersection.

IMPORTANT: Use `openpyxl` and write formulas as strings starting with `=`. Use absolute references to the Data sheet like `Data!$A$21:$A$38` etc. Make sure column/row references match what you discovered in step 0.

Example pattern (adapt ranges after inspection):
```
=INDEX(Data!$B$21:$ZZ$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$ZZ$20, 0))
```

## 3. Write Net patient flow formulas in H35:L40
The Task sheet should have three blocks:
- H12:L17 = Patient Admissions (or similar)
- H19:L24 = Patient Discharges (or similar)
- H26:L31 = Effective Bed Capacity (or similar)

For each of the 6 hospitals (rows 35-40), columns H-L:
```
=(H12-H19)/H26*100
```
Adjust row references so each hospital row in 35-40 maps to the corresponding row offset in the three blocks (row 35→rows 12,19,26; row 36→rows 13,20,27; etc.).

## 4. Write statistics formulas in H42:L47
For each column H through L:
- Row 42: `=MIN(H$35:H$40)`
- Row 43: `=MAX(H$35:H$40)`
- Row 44: `=MEDIAN(H$35:H$40)`
- Row 45: `=AVERAGE(H$35:H$40)`
- Row 46: `=PERCENTILE(H$35:H$40,0.25)`  (use PERCENTILE.INC if PERCENTILE causes #NAME?)
- Row 47: `=PERCENTILE(H$35:H$40,0.75)`  (use PERCENTILE.INC if PERCENTILE causes #NAME?)

CRITICAL: Check the labels in column A/B/C for rows 42-47 to confirm which statistic goes in which row. Match the order to the labels, not my guess above.

NOTE on PERCENTILE vs PERCENTILE.INC: A prior failed task got #NAME? errors. In Excel, `PERCENTILE` should work, but if the verifier expects `PERCENTILE.INC`, use that. Since openpyxl doesn't evaluate formulas, what matters is that the formula is valid Excel. Use `PERCENTILE` (it is valid in modern Excel). If you see labels like "25th percentile" and "75th percentile", use `=PERCENTILE(range, 0.25)` and `=PERCENTILE(range, 0.75)`.

## 5. Write weighted mean formula in H50:L50
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net patient flow using Effective Bed Capacity as weights.

## 6. Save
```python
import os
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## 7. Validate
Reload the saved file and check that:
- Cells H12:L17 contain formula strings (start with '=')
- Cells H35:L40 contain formula strings
- Cells H42:L47 contain formula strings
- Cells H50:L50 contain formula strings
- No cell in the target ranges is None

Print out the formula strings for a sample of cells to confirm correctness.

Then run: `cd /root && python -m pytest tests/ -v` and report results.

## Key Cautions
- Do NOT use `data_only=True` when loading.
- Do NOT evaluate formulas in Python; write them as Excel formula strings.
- Make sure sheet references use the exact sheet name (e.g., `Data!...`).
- After inspection, if the Data sheet year row or code column differs from my examples, adjust ALL formulas accordingly.
- Preserve all existing formatting, merged cells, and content outside the target ranges.
- Do not add sheets, macros, VBA, or helper columns.

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