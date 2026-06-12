# Task Instruction

Execute the following steps to produce /root/output/result.xlsx.

## Step 0 – Inspect the workbook
```python
import openpyxl, os, json
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
ts = wb['Task']
ds = wb['Data']

# Print Task sheet row 10 (header row with years) columns H-L
print('Task row 10 (years):', [ts.cell(row=10, column=c).value for c in range(8,13)])

# Print Task column D rows 12-31 (series codes)
print('Task col D rows 12-31:', [ts.cell(row=r, column=4).value for r in range(12,32)])

# Print Data sheet structure: row 21 headers and a few rows
print('Data row 20 (possible header):', [ds.cell(row=20, column=c).value for c in range(1,30)])
print('Data row 21:', [ds.cell(row=21, column=c).value for c in range(1,30)])
print('Data row 22:', [ds.cell(row=22, column=c).value for c in range(1,30)])
print('Data row 38:', [ds.cell(row=38, column=c).value for c in range(1,30)])

# Check what's in H12 currently
print('H12 current:', ts.cell(row=12, column=8).value)
print('H35 current:', ts.cell(row=35, column=8).value)
print('H42 current:', ts.cell(row=42, column=8).value)
print('H50 current:', ts.cell(row=50, column=8).value)

# Check row labels for rows 42-47
print('Task col A-G rows 42-47:', [[ts.cell(row=r, column=c).value for c in range(1,8)] for r in range(42,48)])

# Check row labels for rows 35-40
print('Task col A-G rows 35-40:', [[ts.cell(row=r, column=c).value for c in range(1,8)] for r in range(35,41)])

# Check row 50
print('Task row 50:', [ts.cell(row=50, column=c).value for c in range(1,13)])

wb.close()
```
Run this and share the full output before proceeding.

## Step 1 – Write lookup formulas in H12:L17, H19:L24, H26:L31

After inspecting the output above, determine:
- The exact row range on Data sheet that holds the lookup table (rows 21:38 per instructions).
- Which column on Data contains the series codes (the lookup key).
- Which row on Data contains the year headers.

Then, for each target cell in the three blocks, write an INDEX/MATCH formula of this pattern:
```
=INDEX(Data!$<first_data_col>$21:$<last_data_col>$38, MATCH($D<row>, Data!$<key_col>$21:$<key_col>$38, 0), MATCH(<year_cell>, Data!$<first_data_col>$<header_row>:$<last_data_col>$<header_row>, 0))
```
Use absolute references for the Data ranges and mixed references ($D for column D, row relative; year cell with column relative, row absolute) so formulas copy correctly across the 5 columns and 6 rows per block.

IMPORTANT: Assign formula strings (starting with '=') to cell `.value` attributes. Verify after assignment that the cells are not None.

## Step 2 – Net reliability gap (H35:L40) and statistics (H42:L47)

For H35:L40, each cell computes:
```
=(<Successful_cell> - <Failed_cell>) / <Capacity_cell> * 100
```
where Successful = H12:L17 block, Failed = H19:L24 block, Capacity = H26:L31 block. The row offsets should align the six regions.

For H42:L47 (column-wise statistics over H35:L40):
- Row 42: `=MIN(H35:H40)` (adjust column for each)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)` — use `PERCENTILE` not `PERCENTILE.INC` to avoid #NAME? errors
- Row 47: `=PERCENTILE(H35:H40,0.75)`

IMPORTANT: Check the row labels (from Step 0 output) to confirm which statistic goes in which row. The order above is a guess — adjust based on actual labels.

## Step 3 – Weighted mean (H50:L50)

For each column c in H..L:
```
=SUMPRODUCT(<H35:H40_col>, <H26:H31_col>) / SUM(<H26:H31_col>)
```
This is the SUMPRODUCT-based weighted mean using Net reliability gap percentages as values and Compute Capacity as weights.

## Step 4 – Save
```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
wb.close()
```

After saving, re-open the file and verify:
1. H12 contains a formula string (not None).
2. H35 contains a formula string.
3. H42 contains a formula string.
4. H50 contains a formula string.
5. No new sheets were added.

Print these verification checks.

## Critical Reminders
- Do NOT use `data_only=True` when loading.
- Do NOT add sheets, macros, VBA, or external links.
- Use `PERCENTILE` (not `PERCENTILE.INC`) for the 25th/75th percentile formulas to avoid #NAME? errors.
- Ensure every formula cell is actually written (not skipped due to a loop bug). After the loop, spot-check a few cells.
- Preserve all existing formatting by not touching cells outside the specified ranges.
- Adapt all row/column references based on the actual inspection output from Step 0. Do not blindly use the template references if the actual structure differs.

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