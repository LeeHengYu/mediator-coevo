# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Phase 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Open /root/data/workbook.xlsx with openpyxl (data_only=False) and inspect:
   - Sheet names (expect 'Task' and 'Data').
   - On sheet 'Data': print rows 20-38, columns A-M, to see where series codes live (likely column A) and where years live (likely row 21, starting from some column). Print enough to identify the exact layout: which column holds series codes, which row holds years, and where data values begin.
   - On sheet 'Task': print rows 9-50, columns D-L, to see the series codes in column D, the years in row 10 (H10:L10), the yellow target ranges, the region labels, the statistic labels in rows 42-47, and row 50.
3. Record the exact Data-sheet layout: series-code column letter, year row number, data start column, data start row. Record the Task-sheet layout: series codes per block, year cells, region names, statistic labels.

## Phase 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three 6-row × 5-column blocks, write an INDEX/MATCH formula of this pattern:

```
=INDEX(Data!<data_range>, MATCH(<series_code_cell>, Data!<series_code_column>, 0), MATCH(<year_cell>, Data!<year_row>, 0))
```

Concrete references (adjust after inspection):
- `<data_range>`: the rectangular block on Data sheet containing the numeric values (e.g., Data!B22:F38 or whatever the inspection reveals).
- `<series_code_column>`: the column of series codes on Data (e.g., Data!A22:A38). Use the same absolute range for every formula.
- `<year_row>`: the row of years on Data (e.g., Data!B21:F21). Use the same absolute range for every formula.
- `<series_code_cell>`: reference to column D of the current row on Task (e.g., $D12 for the first row). Lock the column with $D.
- `<year_cell>`: reference to the year in row 10 of the current column on Task (e.g., H$10 for column H). Lock the row with $10.

Make sure all Data references are absolute ($) so they don't shift. The series-code and year references should be mixed (column-locked or row-locked) so they work when conceptually filled across the block.

Write formulas as strings (do NOT use data_only; keep them as formulas).

## Phase 2 – Net reliability gap in H35:L40
The six regions in rows 35-40 correspond to the same six regions in the three lookup blocks. For each cell, write:

```
=(H12-H19)/H26*100
```

…adjusted so that row 35 uses the data from rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc. Column stays the same (H through L). Use relative references so each cell points to the correct rows in its own column.

Specifically for cell in row r (35-40), column c (H-L):
- Successful API Requests row = r - 23  (i.e., 12-17)
- Failed API Requests row = r - 16  (i.e., 19-24)
- Compute Capacity row = r - 9  (i.e., 26-31)

Formula pattern: `=(<successful_cell>-<failed_cell>)/<capacity_cell>*100`

## Phase 3 – Summary statistics in H42:L47
Read the labels in column D (or wherever they are) for rows 42-47 to determine which statistic goes where. Based on the task description, the six statistics are: minimum, maximum, median, simple mean, 25th percentile, 75th percentile.

For each column c (H through L), the data range is c35:c40. Write:
- MIN row: `=MIN(H35:H40)` (adjust column)
- MAX row: `=MAX(H35:H40)`
- MEDIAN row: `=MEDIAN(H35:H40)`
- AVERAGE row: `=AVERAGE(H35:H40)`
- 25th percentile row: `=PERCENTILE(H35:H40,0.25)` — use PERCENTILE, NOT PERCENTILE.INC or PERCENTILE.EXC, to avoid #NAME? errors in some Excel engines. Actually, use `PERCENTILE.INC` which is the standard replacement. But IMPORTANT: first check if the verifier/test expects PERCENTILE.INC or PERCENTILE. If uncertain, use `PERCENTILE.INC` as it is the modern standard and supported by openpyxl/Excel. If that causes #NAME? errors, fall back to `PERCENTILE`.
- 75th percentile row: `=PERCENTILE.INC(H35:H40,0.75)` (same logic)

IMPORTANT LESSON FROM CROSS-TASK FAILURE: The weighted-campus-energy-balance-calc task failed because percentile formulas produced #NAME? errors. To be safe, use `PERCENTILE.INC` for 25th/75th percentile. This is the standard Excel function. Do NOT use `PERCENTILE.RANK`, `QUARTILE`, or any non-standard name.

## Phase 4 – Weighted mean in H50:L50
For each column c (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
Adjust column letter for each of the 5 columns.

## Phase 5 – Save and validate
1. Save to /root/output/result.xlsx using openpyxl, preserving all existing formatting.
2. Re-open the saved file (data_only=False) and print:
   - A sample of cells from each block (H12, L17, H19, L24, H26, L31) to confirm formulas are present (not None, not values).
   - H35, L40 to confirm net reliability gap formulas.
   - H42:L47 to confirm stat formulas.
   - H50:L50 to confirm SUMPRODUCT formulas.
3. Verify no new sheets were added.
4. Confirm the formulas reference correct cells by spot-checking a couple.

Do NOT use data_only=True when loading for editing. Do NOT add any sheets, macros, VBA, or external links. Do NOT change any existing formatting or cell values outside the specified ranges.

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