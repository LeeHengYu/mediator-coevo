# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
- Open `/root/data/workbook.xlsx` with openpyxl (keep formulas via `data_only=False`).
- Print the sheet names to confirm `Task` and `Data` exist.
- Print the contents of `Task` sheet rows 10-50, columns D through L, to understand the layout: series codes in column D, years in row 10, yellow cell regions, hospital names, etc.
- Print the `Data` sheet rows 21-38 to understand the lookup source structure (column headers, row labels, how series codes and years are organized).
- Pay special attention to:
  - What exactly is in `Task!D12:D17`, `D19:D24`, `D26:D31` (series codes)
  - What exactly is in `Task!H10:L10` (years)
  - How `Data!21:38` is structured: which column has the series code, which row/column has years, and where the values are.
  - Whether Data is organized with series codes in rows and years in columns, or vice versa.

## 2. Determine the correct lookup formula pattern
Based on the Data sheet structure:
- If Data has series codes in a column and years across columns (horizontal), use INDEX/MATCH or VLOOKUP/MATCH.
- If Data has years in a column and series codes across a row (vertical for years), use HLOOKUP/MATCH or INDEX/MATCH.
- Choose INDEX/MATCH as it's most flexible. The formula pattern will be something like:
  `=INDEX(Data!<value_range>, MATCH(<series_code_ref>, Data!<series_code_column>, 0), MATCH(<year_ref>, Data!<year_row>, 0))`
- Adapt the exact references based on what you observe in the Data sheet.

**CRITICAL**: Use absolute references for the lookup arrays (e.g., `Data!$A$21:$A$38`) and mixed references so that the series code reference points to column D of the current row and the year reference points to row 10 of the current column. This allows the formula to work correctly across the grid.

## 3. Populate formulas in H12:L17, H19:L24, H26:L31
Using openpyxl, write the lookup formulas into each cell in these three blocks. For each cell at row `r`, column `c` (where H=8, I=9, J=10, K=11, L=12):
- The series code is in cell `D{r}` on the Task sheet.
- The year is in cell `{col_letter}10` on the Task sheet (where col_letter corresponds to column c).
- Write the INDEX/MATCH formula referencing these two inputs and the Data sheet range rows 21:38.

Make sure:
- Cell references use the correct column letters and row numbers.
- The Data sheet range references are correct based on your inspection.
- Use `$` signs appropriately: lock the lookup arrays, lock column D for series codes (e.g., `$D{r}` or `$D12`), lock row 10 for years (e.g., `H$10`).

## 4. Populate H35:L40 with Net Patient Flow formulas
Net patient flow = (Patient Admissions - Patient Discharges) / Effective Bed Capacity * 100

Based on the layout:
- H12:L17 likely corresponds to one metric (e.g., Patient Admissions)
- H19:L24 likely corresponds to another metric (e.g., Patient Discharges)  
- H26:L31 likely corresponds to another metric (e.g., Effective Bed Capacity)

Verify which block is which by reading the labels (likely in column C or nearby). Then for each cell in H35:L40, write a formula like:
`=(H12-H19)/H26*100` (adjusting row references for each hospital row and using the correct blocks for admissions, discharges, and capacity).

## 5. Populate H42:L47 with summary statistics
For each column (H through L), calculate over the 6 hospital values in rows 35:40:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

**IMPORTANT**: Check the labels in the Task sheet for rows 42-47 to confirm the exact order (min, max, median, mean, 25th, 75th). Match the formula to the label, not to my assumed order.

## 6. Populate H50:L50 with weighted mean
For each column, use SUMPRODUCT with the net patient flow percentages (H35:H40) as values and the effective bed capacity (H26:H31) as weights:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

Use the appropriate column for each cell.

## 7. Save and verify
- Save the workbook to `/root/output/result.xlsx`.
- Reopen it and print sample cells to verify formulas were written correctly.
- Verify no extra sheets were added, no formatting was changed, and all target cells contain formulas (not hardcoded values).
- Print the formula strings from a few cells in each block to confirm correctness.

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