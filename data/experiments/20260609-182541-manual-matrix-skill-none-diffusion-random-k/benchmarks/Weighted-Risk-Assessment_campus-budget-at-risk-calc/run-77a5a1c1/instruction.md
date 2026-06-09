# Task Instruction

Execute the following steps exactly:

## 1. Inspect the workbook
```bash
cd /root && python3 -c "
import openpyxl
wb = openpyxl.load_workbook('data/workbook.xlsx')
for s in wb.sheetnames:
    print('Sheet:', s)
ws = wb['Task']
for r in range(1, 55):
    row_vals = []
    for c in range(1, 15):
        cell = ws.cell(row=r, column=c)
        row_vals.append(f'{cell.coordinate}={cell.value}')
    print(' | '.join(row_vals))
print('--- Data sheet rows 18-40 ---')
wd = wb['Data']
for r in range(18, 41):
    row_vals = []
    for c in range(1, 20):
        cell = wd.cell(row=r, column=c)
        row_vals.append(f'{cell.coordinate}={cell.value}')
    print(' | '.join(row_vals))
print('--- Data sheet row 1-5 (headers) ---')
for r in range(1, 6):
    row_vals = []
    for c in range(1, 20):
        cell = wd.cell(row=r, column=c)
        row_vals.append(f'{cell.coordinate}={cell.value}')
    print(' | '.join(row_vals))
"
```

Examine the output carefully. Identify:
- The series codes in column D for rows 12-17, 19-24, 26-31.
- The years in row 10 for columns H-L.
- The structure of the Data sheet rows 21-38 (which column has series codes, which row has years, where are the values).
- The department names and any labels in column D for rows 35-40.
- The labels in rows 42-47 (min, max, median, mean, 25th percentile, 75th percentile).
- The label in row 50.

## 2. Populate lookup formulas (Step 1)

Using the information gathered, write a Python script that opens the workbook and populates cells H12:L17, H19:L24, and H26:L31 with INDEX/MATCH formulas.

The formula pattern for each cell should be:
```
=INDEX(Data!$<value_start_col>$21:$<value_end_col>$38, MATCH($D<row>, Data!$<series_col>$21:$<series_col>$38, 0), MATCH(H$10, Data!$<header_start_col>$<header_row>:$<header_end_col>$<header_row>, 0))
```

Adjust column letters based on what you discover in the Data sheet. The series code column and the year header row must be identified from inspection. Use absolute references for the Data ranges and mixed references ($D<row> for the series code column, <col>$10 for the year row) so formulas copy correctly across the 5 columns and 6 rows per block.

## 3. Net budget buffer formulas (Step 2)

For H35:L40, the formula is:
```
=(<Committed_Funding_cell> - <Operating_Spend_cell>) / <Approved_Budget_Base_cell> * 100
```
where:
- Committed Funding is in rows 12-17 (the first block)
- Operating Spend is in rows 19-24 (the second block)  
- Approved Budget Base is in rows 26-31 (the third block)

So for cell H35: `=(H12-H19)/H26*100`, H36: `=(H13-H20)/H27*100`, etc.

For H42:L47 summary statistics:
- Row 42 (MIN): `=MIN(H35:H40)` etc.
- Row 43 (MAX): `=MAX(H35:H40)` etc.
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)` etc.
- Row 45 (MEAN): `=AVERAGE(H35:H40)` etc.
- Row 46 (25th percentile): `=_xlfn.PERCENTILE.INC(H35:H40,0.25)`
- Row 47 (75th percentile): `=_xlfn.PERCENTILE.INC(H35:H40,0.75)`

**CRITICAL**: For percentile formulas, you MUST use the `_xlfn.` prefix: `_xlfn.PERCENTILE.INC`. This is required because openpyxl does not automatically add this prefix, and without it the formula will produce a #NAME? error when evaluated. This was the exact failure in the previous attempt.

## 4. Weighted mean (Step 3)

For H50:L50:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
(Repeat for columns I through L.)

## 5. Write the Python script

After inspection, write and run a single Python script that:
1. Loads `/root/data/workbook.xlsx` with openpyxl (data_only=False)
2. Populates all formula cells as described above
3. Saves to `/root/output/result.xlsx`
4. Does NOT change formatting, add sheets, macros, or external links

Make sure to create `/root/output/` directory if it doesn't exist.

## 6. Validate

After saving, re-open the result file and print all formula cells to verify:
- Lookup formulas in H12:L31 use INDEX/MATCH referencing Data sheet
- Net buffer formulas in H35:L40 are correct
- Summary stats in H42:L47 are correct, especially H46:L47 use `_xlfn.PERCENTILE.INC`
- Weighted mean in H50:L50 uses SUMPRODUCT

Also run the test if available:
```bash
cd /root && python3 -m pytest test_output.py -v 2>&1 | head -80
```

If any test fails, read the error, fix the formulas, and re-run.

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