# Task Instruction

Execute the following steps in a single Python script using openpyxl.

## 0 – Inspect the workbook
```python
import openpyxl, shutil, os
os.makedirs('/root/output', exist_ok=True)
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
print('Sheet names:', wb.sheetnames)

# Inspect Task sheet layout
ts = wb['Task']
print('Row 10 (headers):', [ts.cell(r,c).value for r,c in [(10,col) for col in range(1,15)]])
print('Row 11 (sub-headers):', [ts.cell(11,c).value for c in range(1,15)])
for r in range(12,48):
    print(f'Row {r}:', [ts.cell(r,c).value for c in range(1,15)])
print('Row 50:', [ts.cell(50,c).value for c in range(1,15)])

# Inspect Data sheet layout
ds = wb['Data']
for r in range(1,5):
    print(f'Data row {r}:', [ds.cell(r,c).value for c in range(1,20)])
for r in range(19,42):
    print(f'Data row {r}:', [ds.cell(r,c).value for c in range(1,20)])
```
Print everything so we can see exact cell contents, column letters for series codes, year headers, and data layout.

## 1 – Write lookup formulas in H12:L17, H19:L24, H26:L31

For each yellow cell at row `r`, column `c` (H=8 … L=12):
- The series code is in column D of that row on sheet Task: `Task!D{r}`
- The year is in row 10 of that column on sheet Task: `Task!{col_letter}10`
- The data lives on sheet `Data` rows 21:38.

Use INDEX/MATCH pattern. Determine from inspection which column on Data holds the series codes and which row holds the years. Construct formulas like:
```
=INDEX(Data!$B$21:$S$38, MATCH(D12,Data!$A$21:$A$38,0), MATCH(H10,Data!$B$20:$S$20,0))
```
Adjust column/row references based on actual inspection results. The key is:
- MATCH #1 finds the row in the data block matching the series code
- MATCH #2 finds the column in the data block matching the year
- INDEX returns the intersection

## 2 – Write derived formulas in H35:L40

Net production slack = (Finished Output - Scrap And Rework) / Rated Production Capacity * 100

Identify which of the three lookup blocks (H12:L17, H19:L24, H26:L31) corresponds to Finished Output, Scrap And Rework, and Rated Production Capacity by reading labels in the Task sheet (likely around rows 11, 18, 25 or nearby). Then for each cell at row offset `i` (0..5) and column `c`:
```
=(H12-H19)/H26*100
```
(adjusted for actual block assignments and row offsets)

## 3 – Write summary statistics in H42:L47

For each column c in H..L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=_xlfn.PERCENTILE.INC(H35:H40,0.25)`
- Row 47: `=_xlfn.PERCENTILE.INC(H35:H40,0.75)`

**CRITICAL**: For the percentile functions, you MUST use the `_xlfn.` prefix: `_xlfn.PERCENTILE.INC`. This is required by openpyxl for modern Excel functions. Previous attempts failed with #NAME? errors because this prefix was missing.

Check the labels in column D or nearby for rows 42-47 to confirm the order (min, max, median, mean, 25th pct, 75th pct). Adjust row assignments if the labels indicate a different order.

## 4 – Write weighted mean in H50:L50

For each column c in H..L:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean using Net production slack values as the metric and Rated Production Capacity as weights.

## 5 – Save
```python
wb.save('/root/output/result.xlsx')
```

## 6 – Verify
Reload the saved file and print the values/formulas in key cells to confirm formulas were written:
```python
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
ts2 = wb2['Task']
for r in [12,19,26,35,42,46,47,50]:
    print(f'Row {r}:', [ts2.cell(r,c).value for c in range(8,13)])
```

## Important Notes
- Do NOT add any new sheets, macros, VBA, or external links.
- Do NOT modify any existing formatting.
- Use `data_only=False` when loading (default) to preserve existing formulas.
- Write all formulas as strings starting with '='.
- For PERCENTILE functions, always use `_xlfn.PERCENTILE.INC` (with underscore prefix).
- Adjust all cell references based on what you actually see during inspection in step 0. Do not assume — verify the layout first.

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