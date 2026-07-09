# Task Instruction

Execute the following steps precisely to complete the weighted campus energy balance workbook task.

## 0. Inspect the workbook
```bash
pip install openpyxl
```
Then in Python:
```python
import openpyxl
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for name in wb.sheetnames:
 ws = wb[name]
 print(f'=== {name} ===')
 for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column, values_only=False):
 for cell in row:
 if cell.value is not None:
 print(f' {cell.coordinate}: {cell.value}')
```
Read the output carefully. Identify:
- The series codes in column D for rows 12-17, 19-24, 26-31 on sheet `Task`
- The year values in row 10 for columns H through L on sheet `Task`
- The layout of sheet `Data` rows 21-38 (what is in each column, where series codes and years appear)
- The campus names and their row positions
- Any existing formulas or values already present

## 1. Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in the yellow ranges, write a spreadsheet formula (not a Python-computed value) that looks up data from the `Data` sheet rows 21:38. The formula must use two keys:
- The series code from column D of the same row on `Task`
- The year from row 10 of the same column on `Task`

Use INDEX/MATCH or one of the other allowed patterns. The exact formula pattern depends on how the Data sheet is laid out. After inspecting the Data sheet:

- If Data has series codes in a column and years in a row header (i.e., a matrix layout), use something like:
  `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
  Adjust the exact ranges based on what you observe.

- If Data has a tabular/list layout (series code in one column, year in another, value in a third), you may need a two-criteria approach like:
  `=INDEX(Data!value_col, MATCH(1, (Data!code_col=$D12)*(Data!year_col=H$10), 0))`
  entered as an array formula (wrap with curly braces isn't needed in .xlsx; just set the formula).

IMPORTANT: Use absolute references for the Data ranges and mixed references ($D12 for series code column, H$10 for year row) so formulas can be applied across the grid correctly.

Write formulas using openpyxl. For each cell in the range, set `cell.value = '=FORMULA...'`. Do NOT set `cell.data_type` manually; openpyxl handles formula strings starting with '='.

## 2. Net renewable balance in H35:L40

For each campus (rows 35-40) and each year column (H-L), write a formula:
```
=(H12 - H19) / H26 * 100
```
where H12 is the Renewable Generation value, H19 is Grid Consumption, and H26 is Baseline Energy Demand for the same campus and year. Adjust row references based on the actual row mapping:
- Row 35 campus corresponds to row 12, 19, 26
- Row 36 campus corresponds to row 13, 20, 27
- Row 37 → 14, 21, 28
- Row 38 → 15, 22, 29
- Row 39 → 16, 23, 30
- Row 40 → 17, 24, 31

Verify this mapping by checking the campus names/labels in the sheet.

## 3. Summary statistics in H42:L47

For each year column (H through L):
- H42: `=MIN(H35:H40)`
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)` (simple mean)
- H46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- H47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

Check the labels in column D/E/F/G for rows 42-47 to confirm the correct order (min, max, median, mean, 25th, 75th). Adjust row assignments if the labels differ from the order above.

## 4. Weighted mean in H50:L50

For each year column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the net renewable balance percentages using Baseline Energy Demand as weights.

## 5. Save
```python
import shutil, os
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## 6. Verify
Reload the saved file and print all formula cells to confirm:
- H12:L17 contain lookup formulas referencing Data sheet
- H19:L24 contain lookup formulas referencing Data sheet
- H26:L31 contain lookup formulas referencing Data sheet
- H35:L40 contain net balance formulas
- H42:L47 contain MIN/MAX/MEDIAN/AVERAGE/PERCENTILE formulas
- H50:L50 contain SUMPRODUCT formulas
- No extra sheets were added
- Formatting is preserved (spot-check a few cells for fill colors)

## Critical Notes
- You MUST inspect the Data sheet layout before writing any formulas. The exact column/row references in the formulas depend entirely on how Data is structured.
- All values in the yellow cells must be Excel formulas (strings starting with '='), not hardcoded numbers.
- Do not modify any existing content, formatting, or structure. Only fill in the specified cells.
- Do not add sheets, macros, VBA, or external links.
- If the summary statistic labels are in a different order than min/max/median/mean/25th/75th, match the formulas to the actual labels.

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