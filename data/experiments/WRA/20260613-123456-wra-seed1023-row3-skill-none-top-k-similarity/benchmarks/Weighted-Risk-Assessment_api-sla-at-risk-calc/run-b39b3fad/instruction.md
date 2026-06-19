# Task Instruction

Execute the following steps exactly:

## Step 0 – Inspect the workbook
```bash
cp /root/data/workbook.xlsx /root/data/workbook_backup.xlsx
```
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Inspect:
- Sheet `Task`: print rows 9-50 (columns A-L) to understand the layout – especially:
  - Row 10 (the year headers in H10:L10)
  - Column D rows 12-17, 19-24, 26-31 (series codes)
  - Row labels in rows 35-40 (service names for Net SLA buffer)
  - Row labels in rows 42-47 (MIN/MAX/MEDIAN/AVERAGE/PERCENTILE labels)
  - Row 50 label and any weights info
- Sheet `Data`: print rows 20-39 (all columns) to see the lookup table structure – identify:
  - Which row is the header row (row 21 area)
  - How series codes and years are arranged
  - The exact column range of the data table

Print all of this before making any edits.

## Step 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write an INDEX/MATCH formula that:
- Uses the series code from column D of the **same row** (use $D prefix for the column, e.g., `$D12`)
- Uses the year from row 10 of the **same column** (use row-lock, e.g., `H$10`)
- Looks up against the Data sheet rows 21:38

The exact formula pattern (adjust column letters and row numbers based on what you discover in the inspection):
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Make sure:
- The `<data_range>` covers the full rectangular block of numeric data on the Data sheet (rows 21-38, from the first year column to the last year column)
- The `<series_code_column>` is the column on Data that contains the series codes (same row range as data_range)
- The `<year_header_row>` is the header row on Data containing years (same column range as data_range)
- Use absolute references ($) appropriately so the formula can be applied across the H:L columns and down the rows correctly

## Step 2 – Net SLA buffer in H35:L40

Based on the inspection, identify which rows correspond to:
- Block 1 (H12:L17) = e.g., "Latency Budget Preserved"
- Block 2 (H19:L24) = e.g., "Latency Budget Consumed"
- Block 3 (H26:L31) = e.g., "Covered Request Capacity"

For each cell in H35:L40, write the formula:
```
=(H12-H19)/H26*100
```
Adjusted so that row 35 references rows 12, 19, 26; row 36 references rows 13, 20, 27; etc. Lock the column with $ on the column letter if needed, but since each cell is independent, just use relative references that map correctly.

## Step 3 – Statistics in H42:L47

For each column H through L, based on the labels discovered in column D/E of rows 42-47, write:
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- AVERAGE (simple mean): `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)`  (**use PERCENTILE, not PERCENTILE.INC or PERCENTILE.EXC** – the failed run got #NAME? errors from using unsupported function names)
- 75th percentile: `=PERCENTILE(H35:H40,0.75)`

IMPORTANT: Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC`. The dotted variants cause #NAME? errors in openpyxl/Excel compatibility.

Match each formula to the correct row based on the actual label text you find in the workbook.

## Step 4 – Weighted mean in H50:L50

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net SLA buffer percentages weighted by Covered Request Capacity.

## Step 5 – Save
```python
import os
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## Step 6 – Verify
Reopen `/root/output/result.xlsx` with openpyxl (data_only=False). Print the formulas in:
- A sample cell from each lookup block (e.g., H12, H19, H26)
- H35, H40 (Net SLA buffer)
- H42:H47 (statistics)
- H50 (weighted mean)

Confirm no cells contain #NAME?, None, or empty values where formulas are expected. Confirm the formula syntax uses only standard Excel functions: INDEX, MATCH, MIN, MAX, MEDIAN, AVERAGE, PERCENTILE, SUMPRODUCT, SUM.

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