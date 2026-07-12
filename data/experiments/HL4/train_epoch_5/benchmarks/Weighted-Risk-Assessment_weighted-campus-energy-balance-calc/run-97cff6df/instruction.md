# Task Instruction

Execute the following steps in order.

## 0 – Inspect the workbook
```bash
cp /root/data/workbook.xlsx /root/output/result.xlsx
```
Open `/root/output/result.xlsx` with `openpyxl` (with `data_only=False`) and inspect:
- Sheet `Task`: print rows 10-50 (columns D through L) so you can see the series codes in column D, the years in row 10, and the layout of the yellow target cells.
- Sheet `Data`: print rows 21-38 (all columns) to see the source data layout – column headers, row labels, and where values live.

Pay special attention to:
1. The exact series codes in column D of the Task sheet for rows 12-17, 19-24, 26-31, and 35-40.
2. The exact years in H10:L10.
3. The Data sheet structure: which column holds the series code, which row holds the year headers, and where the numeric data starts.
4. The campus names in the Task sheet rows 35-40 and rows 12-17/19-24/26-31.
5. Any existing content in the target cells (H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50).

Print everything clearly before proceeding.

## 1 – Build the lookup formulas for H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write an `INDEX/MATCH` formula (or `XLOOKUP` with `MATCH` – pick one pattern and use it consistently). The formula must:
- Look up the series code from column D of the *current row* on the Task sheet.
- Look up the year from row 10 of the *current column* on the Task sheet.
- Search in the Data sheet rows 21:38.
- Use the series-code column and year-header row from the Data sheet that you identified in step 0.

Make sure all sheet references use the exact sheet name (e.g., `Data!...`). Use absolute references (`$`) where appropriate so the formula is correct for each cell position.

Write these formulas with openpyxl by assigning formula strings to each cell.

## 2 – Net renewable balance formulas in H35:L40

For each campus (rows 35-40) and each year (columns H-L), write a formula:
```
= (RenewableGeneration - GridConsumption) / BaselineEnergyDemand * 100
```
where:
- RenewableGeneration = the corresponding cell in the H12:L17 block (same campus row offset, same column)
- GridConsumption = the corresponding cell in the H19:L24 block
- BaselineEnergyDemand = the corresponding cell in the H26:L31 block

Verify the row mapping: the first campus in rows 12, 19, 26, 35 should all be the same campus. Print the campus labels from column D (or whichever column has them) for rows 12-17, 19-24, 26-31, 35-40 to confirm alignment.

## 3 – Summary statistics in H42:L47

For each year column (H through L), write formulas in rows 42-47:
- Row 42: `=MIN(H35:H40)` (column-wise minimum)
- Row 43: `=MAX(H35:H40)` (column-wise maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

Check the Task sheet labels in column D for rows 42-47 to confirm which row is which statistic. The order above is a guess – use the actual labels to assign the correct formula to each row.

## 4 – Weighted mean in H50:L50

For each year column (H through L), write a `SUMPRODUCT` formula:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the net renewable balance percentages using Baseline Energy Demand as weights.

## 5 – Save and verify

Save the workbook to `/root/output/result.xlsx`.

Then reopen it with openpyxl and print the formula content of a sample of cells from each block (e.g., H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50) to confirm formulas were written correctly.

Do NOT:
- Add any new sheets, macros, VBA, external links, or helper tabs.
- Change any existing formatting.
- Delete or modify any existing content outside the target cells.
- Use Python-computed numeric values; all target cells must contain Excel formula strings.

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