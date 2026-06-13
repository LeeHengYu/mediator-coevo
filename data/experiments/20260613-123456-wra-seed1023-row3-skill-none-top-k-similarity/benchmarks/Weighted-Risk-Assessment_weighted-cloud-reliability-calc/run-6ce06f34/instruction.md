# Task Instruction

Execute the following steps carefully to produce /root/output/result.xlsx.

## Step 0 – Inspect the workbook
```python
import openpyxl, os, pprint
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for name in wb.sheetnames:
    print(f'--- Sheet: {name} ---')
    ws = wb[name]
    print(f'  Dimensions: {ws.dimensions}')
```
Then inspect:
- **Task sheet**: Print rows 10-50, columns A-L (values). Pay special attention to:
  - Row 10 (year headers in H10:L10)
  - Column D rows 12-17, 19-24, 26-31 (series codes)
  - Row labels in rows 35-47 (what statistics are expected: check exact text for Min, Max, Median, Mean/Average, 25th percentile, 75th percentile)
  - Row 50 label
- **Data sheet**: Print rows 1-5 and rows 18-40, all columns. Identify:
  - Which row contains the header row for the data block (row 20 or 21?)
  - Where the series codes appear (column A? B?)
  - Where the year columns start
  - The exact range of data rows 21:38

Print all of this before writing any formulas.

## Step 1 – Populate lookup blocks (H12:L17, H19:L24, H26:L31)

Based on the inspection, write INDEX/MATCH formulas. The pattern for each cell should be:
```
=INDEX(Data!$<data_range>, MATCH($D12, Data!$<series_code_column>, 0), MATCH(H$10, Data!$<year_header_row>, 0))
```
Where:
- `$D12` is the series code in column D of the current row (lock column with $D)
- `H$10` is the year in row 10 (lock row with $10)
- `<data_range>` is the rectangular block of numeric data on the Data sheet (rows 21:38, from the first year column to the last year column)
- `<series_code_column>` is the column on Data sheet containing the series codes (same rows 21:38)
- `<year_header_row>` is the row on Data sheet containing the year headers (same columns as data)

Use absolute references for the Data sheet ranges so formulas can be filled across all 90 cells (6 rows × 5 columns × 3 blocks).

Write formulas to all cells in H12:L17, H19:L24, H26:L31 using nested loops.

## Step 2 – Net reliability gap (H35:L40) and statistics (H42:L47)

For H35:L40, write formulas computing:
```
=(H12 - H19) / H26 * 100
```
Adjust row references for each of the 6 regions (rows 35-40 correspond to rows 12-17, 19-24, 26-31 respectively).

For H42:L47, read the exact row labels in column A/B/C/D/E/F/G for rows 42-47 to determine the order of statistics. Then write column-wise formulas over H35:H40 (through L35:L40):
- **Minimum**: `=MIN(H35:H40)`  (or `=MIN(H$35:H$40)` with locked rows)
- **Maximum**: `=MAX(H35:H40)`
- **Median**: `=MEDIAN(H35:H40)`
- **Mean**: `=AVERAGE(H35:H40)`
- **25th percentile**: `=PERCENTILE(H35:H40,0.25)` — use `PERCENTILE` (NOT `PERCENTILE.INC` or `PERCENTILE.EXC`, as these caused #NAME? errors in prior runs)
- **75th percentile**: `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Match the row label to the correct function. Read the labels first. If the label says "Mean" use AVERAGE. If it says "25th Percentile" use PERCENTILE with 0.25. Do NOT use PERCENTILE.INC or PERCENTILE.EXC — use the legacy `PERCENTILE` function name only.

## Step 3 – Weighted mean (H50:L50)

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net reliability gap using Compute Capacity as weights.

## Step 4 – Save
```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## Step 5 – Verify
Reload the saved file and print the formula values in key cells:
- H12, L17 (lookup block)
- H35, L40 (net reliability gap)
- H42:H47 (statistics column H)
- H50 (weighted mean)

Confirm no cell contains None or starts with an error pattern. All should be formula strings starting with '='.

## Important Constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Use `data_only=False` when loading (default) to preserve existing formulas.
- Use the legacy function names: `PERCENTILE` (not `.INC`/`.EXC`), `AVERAGE`, `MEDIAN`, `MIN`, `MAX`, `SUMPRODUCT`, `SUM`, `INDEX`, `MATCH`.

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