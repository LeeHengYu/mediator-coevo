# Task Instruction

Execute the following steps in order.

## Step 0 – Inspect the workbook
```python
import openpyxl, shutil, os

wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for name in wb.sheetnames:
    print(f'--- Sheet: {name} ---')
    ws = wb[name]
    print(f'  Dimensions: {ws.dimensions}')
    print(f'  Max row: {ws.max_row}, Max col: {ws.max_column}')

# Print Task sheet structure
ts = wb['Task']
print('\n=== Task sheet – rows 1-55, cols A-M ===')
for row in ts.iter_rows(min_row=1, max_row=55, min_col=1, max_col=13, values_only=False):
    vals = [(c.coordinate, c.value) for c in row if c.value is not None]
    if vals:
        print(vals)

# Print Data sheet – especially rows 18-40 and headers
ds = wb['Data']
print('\n=== Data sheet – rows 1-5 (headers) ===')
for row in ds.iter_rows(min_row=1, max_row=5, min_col=1, max_col=20, values_only=False):
    vals = [(c.coordinate, c.value) for c in row if c.value is not None]
    if vals:
        print(vals)

print('\n=== Data sheet – rows 18-40 ===')
for row in ds.iter_rows(min_row=18, max_row=40, min_col=1, max_col=20, values_only=False):
    vals = [(c.coordinate, c.value) for c in row if c.value is not None]
    if vals:
        print(vals)

wb.close()
```
Print everything; do NOT skip this step. You need the exact layout to write correct formulas.

## Step 1 – Identify the mapping
From the printout, determine:
- The series codes in column D of rows 12-17, 19-24, 26-31 on sheet `Task`.
- The years in row 10 for columns H through L on sheet `Task`.
- On sheet `Data`, rows 21-38: which column holds the series codes, which row holds the years, and where the numeric data lives.
- Confirm the exact cell references for the lookup range on `Data`.

## Step 2 – Write the lookup formulas (H12:L17, H19:L24, H26:L31)
Using openpyxl (keeping the workbook open without data_only), write INDEX/MATCH/MATCH formulas into every cell in those three blocks. The pattern for each cell should be:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```
Adjust the exact ranges based on what you discovered in Step 0. Use `$D` (absolute column) and `$10` (absolute row) references so formulas shift correctly across the grid. Loop over the 6 rows × 5 columns for each block.

## Step 3 – Net renewable balance (H35:L40)
For each of the 6 campus rows (rows 35-40) and 5 year columns (H-L), write a formula:
```
=(H12 - H19) / H26 * 100
```
where the row references correspond to:
- Renewable Generation block: rows 12-17
- Grid Consumption block: rows 19-24
- Baseline Energy Demand block: rows 26-31
So row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.

## Step 4 – Summary statistics (H42:L47)
For each column H through L, write formulas in rows 42-47. First confirm from the Task sheet printout the exact labels and their order (e.g., MIN, MAX, MEDIAN, AVERAGE, PERCENTILE.INC 25th, PERCENTILE.INC 75th – or whatever order the labels show). Then write the corresponding formulas referencing H35:H40 (etc.):
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- AVERAGE: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE.INC(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE.INC(H35:H40,0.75)`

Match the row to the label exactly as shown on the sheet.

## Step 5 – Weighted mean (H50:L50)
For each column H through L, write:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net renewable balance percentages using Baseline Energy Demand as weights.

## Step 6 – Save
```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
wb.close()
```

## Step 7 – Verify
Reopen `/root/output/result.xlsx` with openpyxl and print the formula strings (not computed values) for a sample of cells:
- H12, L17 (lookup block 1)
- H19, L24 (lookup block 2)
- H26, L31 (lookup block 3)
- H35, L40 (net balance)
- H42, H47 (summary stats)
- H50, L50 (weighted mean)

Confirm every cell contains a formula string (starts with '='). If any cell is None or a plain number, fix it before finishing.

IMPORTANT:
- Do NOT compute any values in Python. Every result cell must contain an Excel formula string.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting (fonts, fills, borders, etc.).
- Use the exact sheet name 'Data' in cross-sheet references.

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