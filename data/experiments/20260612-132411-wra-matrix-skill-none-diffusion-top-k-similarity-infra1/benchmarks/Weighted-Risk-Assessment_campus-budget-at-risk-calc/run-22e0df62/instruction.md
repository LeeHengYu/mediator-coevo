# Task Instruction

Execute the following steps to produce /root/output/result.xlsx:

## 0 – Preparation
```bash
mkdir -p /root/output
pip install openpyxl --quiet
```

## 1 – Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Print:
- Sheet names.
- `Task` sheet: contents of column D rows 12-17, 19-24, 26-31 (series codes), row 10 columns H-L (years), rows 35-40 labels in column D, rows 42-47 labels in column C or D, row 50 label.
- `Data` sheet: row 21 (header row) and a sample of rows 22-38, focusing on the layout – is the data organized with series codes in a column and years across columns, or vice versa? Identify which column holds the series code and which row holds the year headers.

Print all of this so you understand the exact geometry before writing any formulas.

## 2 – Write the lookup formulas (Step 1)
Using openpyxl, write INDEX/MATCH formulas into the yellow cells.

For each cell in ranges H12:L17, H19:L24, H26:L31:
- The series code is in column D of that row (e.g., D12 for row 12).
- The year is in row 10 of that column (e.g., H10 for column H).
- The data lives on sheet `Data` rows 21:38.

Based on your inspection of the Data sheet layout, construct the correct INDEX/MATCH formula. The typical pattern when data has series codes in a column (say column A or B) and years in a header row (row 21) is:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```
Adjust the ranges based on what you actually find. Use absolute references for the lookup arrays ($D12 for series code, H$10 for year) so formulas copy correctly across the grid.

Write these formulas as strings into each cell. Do NOT set cell values to numbers.

## 3 – Net budget buffer formulas (Step 2, rows 35-40)
Identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to:
- Committed Funding
- Operating Spend  
- Approved Budget Base

by reading the labels near rows 11, 18, 25 on the Task sheet.

Then for each cell in H35:L40, write a formula:
```
=(committed_funding_cell - operating_spend_cell) / approved_budget_base_cell * 100
```
where the three referenced cells are in the same column and correspond to the same department (same relative row offset within each block).

## 4 – Summary statistics (rows 42-47)
Read the labels in rows 42-47 from the Task sheet to determine the exact order of: MIN, MAX, MEDIAN, AVERAGE (mean), 25th percentile, 75th percentile.

For each column H through L, write the appropriate Excel formula:
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- AVERAGE: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40, 0.25)` — use PERCENTILE, NOT PERCENTILE.INC or PERCENTILE.EXC
- 75th percentile: `=PERCENTILE(H35:H40, 0.75)` — use PERCENTILE, NOT PERCENTILE.INC or PERCENTILE.EXC

Place each formula in the row that matches its label.

## 5 – Weighted mean (row 50)
For each column H through L, write in row 50:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the net budget buffer percentages as values and the Approved Budget Base block (H26:L31) as weights.

## 6 – Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change any existing formatting, do NOT add sheets, macros, VBA, or external links.

## 7 – Validate
Reopen the saved file with openpyxl (data_only=False) and print:
- A sample formula from each block (H12, H19, H26, H35, H42-H47, H50) to confirm they are formula strings, not bare values.
- Confirm no new sheets were added.
- Confirm the file exists and is non-empty.

## Critical Reminders
- Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC` (these cause #NAME? errors).
- All formulas must be strings starting with `=`.
- Inspect the actual Data sheet layout before constructing INDEX/MATCH; do not assume column/row positions.
- Match the summary statistic labels exactly to the correct rows 42-47.
- The Approved Budget Base block identity must be confirmed from sheet labels, not assumed.

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