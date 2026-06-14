# Task Instruction

Execute the following steps exactly, in order.

## 0 – Setup
```python
import shutil, os
os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')
```

## 1 – Inspect the workbook
Open `/root/output/result.xlsx` with openpyxl (data_only=False).

Print:
- Sheet names.
- On sheet `Task`: the contents of every cell in rows 10-50 for columns D through L (use `cell.value`). Pay special attention to:
  - Row 10 (year headers in H10:L10).
  - Column D rows 12-17, 19-24, 26-31 (series codes).
  - Row 35 label and rows 35-40 column D (region names for net reliability gap).
  - Rows 42-47 column D or G (statistic labels: min, max, median, mean, 25th pctl, 75th pctl – note exact row-to-label mapping).
  - Row 50 column D or G (GCM weighted mean label).
- On sheet `Data`: print rows 19-40 fully (all columns with data) to understand the layout – column letters, header row, series codes column, and year columns.

Do NOT proceed until you have printed and read all of this. You need the exact column letters and row numbers.

## 2 – Determine the lookup structure
From the inspection, identify:
- Which column on `Data` contains the series codes (let's call it `DATA_KEY_COL`).
- Which row on `Data` contains the year headers (let's call it `DATA_YEAR_ROW`).
- The data range on `Data` that spans rows 21:38 (as stated in the task).
- The exact years in H10:L10 on `Task`.

## 3 – Write the Python script to populate formulas
Use openpyxl to open `/root/output/result.xlsx` (data_only=False). Populate formulas as follows.

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write an INDEX/MATCH formula:
```
=INDEX(Data!<data_values_range>, MATCH($D{row}, Data!<key_column_range>, 0), MATCH(H$10, Data!<year_header_range>, 0))
```
Replace the placeholders with the actual ranges found in Step 1. Use `$D{row}` (column-absolute) and `H$10` style (row-absolute, but column relative so it shifts from H to L). Make sure the ranges are correct and use absolute references where needed (e.g., the data range and key column should be absolute so they don't shift).

### Step 2 – Net reliability gap in H35:L40
The formula is: `(Successful API Requests - Failed API Requests) / Compute Capacity * 100`

From the inspection, determine which of the three lookup blocks (H12:L17, H19:L24, H26:L31) corresponds to each indicator. Map the six rows in each block to the six regions, and the six rows in H35:L40 to the same six regions.

For each cell in H35:L40, write a formula like:
```
=(H12 - H19) / H26 * 100
```
(Adjust row references so that each row in 35-40 uses the corresponding rows from the three blocks for the same region.)

### Step 2 continued – Statistics in H42:L47
Map the exact statistic labels (from your inspection of column D/G rows 42-47) to formulas. For each column H through L:
- MIN: `=MIN(H35:H40)` (or whichever row range covers the 6 regions)
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- MEAN (simple): `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Use `PERCENTILE` (legacy name), NOT `PERCENTILE.INC`. The previous attempt failed because the function name was not recognized. `PERCENTILE` is the safe, universally compatible choice.

Assign each formula to the correct row based on the label you observed. Do NOT assume the order – verify from your inspection.

### Step 3 – Weighted mean in H50:L50
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the net reliability gap percentages using Compute Capacity as weights.

## 4 – Save and verify
Save the workbook to `/root/output/result.xlsx`.

Then reopen it (data_only=False) and print the `.value` of every cell in:
- H12:L17 (should be formula strings starting with '=')
- H19:L24
- H26:L31
- H35:L40
- H42:L47
- H50:L50

Confirm:
1. No cell is None.
2. All cells contain formula strings.
3. The PERCENTILE rows use `PERCENTILE(` not `PERCENTILE.INC(`.
4. The INDEX/MATCH formulas reference the correct Data sheet ranges.
5. The SUMPRODUCT formula is in row 50.

If any cell is None or incorrect, fix it before finishing.

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