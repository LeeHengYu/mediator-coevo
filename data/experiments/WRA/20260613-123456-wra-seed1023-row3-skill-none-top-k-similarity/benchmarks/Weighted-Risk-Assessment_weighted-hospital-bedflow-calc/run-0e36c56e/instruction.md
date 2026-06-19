# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0. Inspect the workbook
```python
import openpyxl, pprint
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for s in wb.sheetnames:
    print(f'--- {s} ---')
    ws = wb[s]
    print(f'  rows: {ws.min_row}-{ws.max_row}, cols: {ws.min_column}-{ws.max_column}')
```
Then dump:
- Sheet `Task`: rows 1-55, columns A-L (especially column D for series codes, row 10 for years, and the yellow target ranges).
- Sheet `Data`: rows 1-40, all used columns (especially rows 21-38 to understand the lookup table layout — which column holds the series code key, which row holds years, and where the numeric data lives).

Print every cell value so you can see the exact layout. Do NOT guess.

## 1. Understand the Data sheet layout
After inspecting, identify:
- The column in `Data` rows 21-38 that contains the series code (the lookup key).
- The row in `Data` that contains the year headers (so MATCH can find the correct column).
- The data range for VLOOKUP / INDEX-MATCH.

## 2. Write lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these ranges, write an Excel formula using `INDEX` + `MATCH`. The pattern should be:
```
=INDEX(Data!<data_columns>, MATCH($D<row>, Data!<key_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```
Adjust the exact references based on what you found in step 0/1. Make sure:
- `$D<row>` references the series code in column D of the current row on the Task sheet (use $ to lock the column).
- `H$10` (or I$10, J$10, etc.) references the year in row 10 (use $ to lock the row).
- The Data ranges are absolute references (e.g., `Data!$A$21:$A$38` for keys, `Data!$B$20:$F$20` for year headers — adjust to actual columns).
- Every formula string starts with `=`.

Write formulas using openpyxl: `ws.cell(row=r, col=c).value = '=INDEX(...)'`

## 3. Write Net patient flow formulas in H35:L40
For each of the 6 hospitals (rows 35-40), compute:
```
=(H12 - H19) / H26 * 100
```
where H12 is Patient Admissions, H19 is Patient Discharges, H26 is Effective Bed Capacity for the same hospital and same column. Adjust row references for each hospital (offsets 0-5). Use relative column references so they shift across H-L.

Write these as Excel formulas.

## 4. Write summary statistics in H42:L47
For columns H through L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)` — use `PERCENTILE` not `PERCENTILE.INC` to avoid #NAME? errors in some Excel engines.
- Row 47: `=PERCENTILE(H35:H40,0.75)`

IMPORTANT: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid #NAME? errors.

## 5. Write weighted mean in H50:L50
For each column (H through L):
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

## 7. Verify
Reload the saved file and print the values of key cells (H12, H35, H42, H46, H47, H50) to confirm they contain formula strings starting with `=`. Also print the Data sheet layout summary to confirm your MATCH ranges were correct.

## Critical reminders
- Do NOT add sheets, macros, VBA, or helper tabs.
- Do NOT change existing formatting.
- Every target cell must contain a formula string (starting with `=`), not a Python-computed number.
- Inspect the actual workbook thoroughly before writing any formulas. The exact row/column layout of the Data sheet is essential.
- If any cell ends up with a Python None or a hardcoded number instead of a formula, the task fails.
- After saving, re-read the file and verify formulas are present.

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