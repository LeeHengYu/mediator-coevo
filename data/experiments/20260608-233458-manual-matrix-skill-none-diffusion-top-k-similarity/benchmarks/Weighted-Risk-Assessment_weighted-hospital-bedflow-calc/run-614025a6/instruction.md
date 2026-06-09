# Task Instruction

Execute the following steps carefully and in order.

## 0. Inspect the Data sheet layout

```python
import openpyxl, shutil, os
os.makedirs('/root/output', exist_ok=True)
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
```

Before writing any formulas, read and print:
1. The entire `Data` sheet rows 19-40 (all columns A-Z or until empty). Print each row as a list so you can see where the series codes live, where the years live, and how the data table is oriented.
2. The `Task` sheet rows 1-55 (columns A-L). Print each row. Pay special attention to:
   - Column D rows 12-17, 19-24, 26-31 (series codes)
   - Row 10 columns H-L (years)
   - The labels in rows 35-40, 42-47, 50

Do NOT proceed to write formulas until you have printed and understood both sheets.

## 1. Determine the INDEX/MATCH formula structure

From the Data sheet inspection, determine:
- Is the data table arranged with series codes in a column and years in a row, or vice versa?
- Which exact column contains the series codes on the Data sheet? (e.g., column A, B, C…)
- Which exact row contains the years on the Data sheet?
- What is the data range (the rectangle of numeric values)?

Based on this, construct an INDEX/MATCH formula. The general pattern is:
```
=INDEX(data_range, MATCH(series_code_ref, series_code_column, 0), MATCH(year_ref, year_row, 0))
```
where:
- `data_range` is the rectangle of values on `Data` sheet (rows 21:38, appropriate columns)
- `series_code_ref` is the cell in column D of the current row on `Task` sheet
- `series_code_column` is the column of series codes on `Data` sheet (same rows as data_range)
- `year_ref` is the cell in row 10 on `Task` sheet (column H, I, J, K, or L)
- `year_row` is the row of years on `Data` sheet (same columns as data_range)

All Data sheet references must use the sheet prefix `Data!` and use `$` signs appropriately:
- The series code column reference should have its column locked (e.g., `Data!$B$21:$B$38`)
- The year row reference should have its row locked (e.g., `Data!$C$20:$R$20` — adjust to actual)
- The data range should be fully locked (e.g., `Data!$C$21:$R$38` — adjust to actual)
- The series code cell reference (`$D12`) should lock the column but not the row
- The year cell reference (`H$10`) should lock the row but not the column

## 2. Write lookup formulas into H12:L17, H19:L24, H26:L31

Using openpyxl, set the `.value` of each cell to the formula string. Loop over the three blocks (rows 12-17, 19-24, 26-31) and columns H-L (columns 8-12). For example:
```python
for row in list(range(12,18)) + list(range(19,25)) + list(range(26,32)):
    for col in range(8, 13):  # H=8, I=9, J=10, K=11, L=12
        formula = '=INDEX(Data!$C$21:$R$38,MATCH($D{row},Data!$B$21:$B$38,0),MATCH({col_letter}$10,Data!$C$20:$R$20,0))'.format(row=row, col_letter=openpyxl.utils.get_column_letter(col))
        ws_task.cell(row=row, column=col).value = formula
```
Adjust the exact Data sheet ranges based on your inspection in Step 0.

## 3. Write Net Patient Flow formulas in H35:L40

The formula for each cell is:
```
=(HXX_admissions - HXX_discharges) / HXX_capacity * 100
```
where admissions are in rows 12-17, discharges in rows 19-24, and capacity in rows 26-31. The six hospitals map as:
- Row 35 uses rows 12, 19, 26
- Row 36 uses rows 13, 20, 27
- Row 37 uses rows 14, 21, 28
- Row 38 uses rows 15, 22, 29
- Row 39 uses rows 16, 23, 30
- Row 40 uses rows 17, 24, 31

For cell H35: `=(H12-H19)/H26*100`

Loop and set these formulas.

## 4. Write statistics formulas in H42:L47

For each column (H through L):
- Row 42 (Min): `=MIN(H35:H40)`
- Row 43 (Max): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

IMPORTANT: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) — the cross-task feedback confirms that `.INC`/`.EXC` variants cause `#NAME?` errors in the evaluation environment.

## 5. Write weighted mean formula in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```

## 6. Verify the row labels match expectations

Before saving, print the Task sheet rows 35-50 column A-D to confirm that:
- Rows 35-40 are the six hospitals
- Row 42 is Min, 43 is Max, 44 is Median, 45 is Mean, 46 is 25th percentile, 47 is 75th percentile
- Row 50 is the MHN weighted mean

If the labels don't match these row assignments, adjust the formula row numbers accordingly.

## 7. Save

```python
wb.save('/root/output/result.xlsx')
```

Do NOT add any new sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

## 8. Quick validation

Reload the saved file and print the formula strings in cells H12, H35, H42, H46, H50 to confirm they are set correctly and are not None.

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