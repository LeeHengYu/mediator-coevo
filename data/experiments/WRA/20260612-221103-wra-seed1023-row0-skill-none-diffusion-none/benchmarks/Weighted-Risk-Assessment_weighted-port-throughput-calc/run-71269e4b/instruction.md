# Task Instruction

Execute the following steps in a single Python script using openpyxl.

## 0. Inspect the workbook structure
```python
import openpyxl, os, shutil

os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')

wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws_task = wb['Task']
ws_data = wb['Data']

# Print Task sheet layout: columns A-L, rows 1-55
for r in range(1, 56):
    vals = []
    for c in range(1, 13):  # A=1 .. L=12
        v = ws_task.cell(row=r, column=c).value
        vals.append(str(v) if v is not None else '')
    print(f'Row {r:>2}: {vals}')

print('\n--- Data sheet rows 1-5 and 18-40 ---')
for r in list(range(1, 6)) + list(range(18, 41)):
    vals = []
    for c in range(1, 20):
        v = ws_data.cell(row=r, column=c).value
        vals.append(str(v) if v is not None else '')
    print(f'Data Row {r:>2}: {vals}')
```
Run this first, read the output carefully, then proceed.

## 1. Identify key layout elements from the inspection output
- **Task sheet row 10**: contains the year headers in columns H-L.
- **Task sheet column D**: contains the series codes for each row in the three lookup blocks (H12:L17, H19:L24, H26:L31).
- **Data sheet rows 21-38**: the source data table. Identify which row contains the header row (likely row 21 with series codes in a column and years across columns, or transposed). Note the exact column that holds series codes and the row that holds years.

## 2. Write lookup formulas (Step 1)
For every cell in H12:L17, H19:L24, and H26:L31, write an INDEX/MATCH/MATCH formula:
```
=INDEX(Data!<data_range>, MATCH($D{row}, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```
Adjust the exact ranges based on what you observed:
- `<data_range>`: the rectangular block of numeric values on the Data sheet (e.g., Data!$B$22:$F$38 or similar).
- `<series_code_column>`: the column of series codes on Data (e.g., Data!$A$22:$A$38).
- `<year_header_row>`: the row of year values on Data (e.g., Data!$B$21:$F$21).

Use absolute references for the data range and lookup vectors; use $D{row} (absolute column, relative row) and H$10 (relative column, absolute row) so the formula copies correctly across the grid.

Write the formula string into each cell using `ws_task.cell(row=r, column=c).value = formula_string`.

## 3. Write Net container flow formulas (Step 2, rows 35-40)
For each port (6 rows), the formula is:
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to the Loaded Containers Inbound row for that port, H19 to Loaded Containers Outbound, and H26 to Terminal Throughput Capacity. Map port rows: row 35 uses rows 12, 19, 26; row 36 uses 13, 20, 27; etc. through row 40 using rows 17, 24, 31. Adjust column letters H-L accordingly.

## 4. Write summary statistics (Step 2, rows 42-47)
Read the labels in column A/B/C/D for rows 42-47 to determine which statistic goes in which row. Then for each column H-L, write:
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- MEAN (simple average): `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)`

Match each formula to the correct row based on the label you read.

## 5. Write weighted mean (Step 3, row 50)
For each column H-L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the net-flow percentages using Terminal Throughput Capacity as weights.

## 6. Save
```python
wb.save('/root/output/result.xlsx')
```

## 7. Verify
Reload the saved file and print cells in the formula regions to confirm they contain formula strings (starting with '='). Check that no cell is None.

## Critical reminders
- Do NOT skip the save step.
- Do NOT add new sheets, macros, or VBA.
- Do NOT alter existing formatting.
- Use `ws_task.cell(row=r, column=c).value = '=FORMULA...'` to write formulas.
- All formula references to the Data sheet must be prefixed with `Data!`.
- Double-check row/column mappings from the inspection output before writing any formula.

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