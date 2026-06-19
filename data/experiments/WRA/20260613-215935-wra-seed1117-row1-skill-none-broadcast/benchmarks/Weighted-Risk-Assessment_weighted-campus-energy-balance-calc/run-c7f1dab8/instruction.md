# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 – Environment & Inspection
```bash
mkdir -p /root/output
pip install openpyxl --quiet
```
Open `/root/data/workbook.xlsx` with openpyxl and inspect:
- Sheet `Task`: print rows 10-11 (header row with years), rows 12-31 (lookup blocks with series codes in column D), rows 35-50 (derived formulas area). Note exact column letters and content.
- Sheet `Data`: print rows 21-38 to understand the data layout (which row holds which series code, which columns hold which years).
- Print the exact values in `D12:D17`, `D19:D24`, `D26:D31` on `Task` to know the series codes.
- Print the exact values in row 10 columns H-L on `Task` to know the year headers.
- Print the header row and first data row of the `Data` sheet rows 21-38 to understand orientation (series codes in a column? years in a row?).

This inspection is critical. Do NOT skip it.

## 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31

Use `INDEX/MATCH` pattern. For each cell at row `r`, column `c` (H=8, I=9, …, L=12):

```
=INDEX(Data!$A$21:$Z$38, MATCH($D{r}, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$A$21:$Z$21, 0))
```

Adjust the ranges based on what you find during inspection:
- The first MATCH searches for the series code (`$D{r}`) in the leftmost column of the Data block.
- The second MATCH searches for the year (`H$10`, `I$10`, etc.) in the header row of the Data block.
- Use absolute references for the data range and header row/column; use `$D{r}` (column-absolute) for the series code and `{col}$10` (row-absolute) for the year.

Fill all 18 cells in each of the three 6×5 blocks.

## 2 – Net renewable balance in H35:L40

For each campus row `i` (0-5), the formula in cell `(35+i, c)` should be:
```
=({cell from H12+i block} - {cell from H19+i block}) / {cell from H26+i block} * 100
```
Concretely, for row 35 column H:
```
=(H12 - H19) / H26 * 100
```
For row 36 column H:
```
=(H13 - H20) / H27 * 100
```
And so on for all 6 rows × 5 columns.

## 3 – Summary statistics in H42:L47

For each column `c` (H through L):
- Row 42: `=MIN(H35:H40)` (adjust column letter)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

Check the labels in column D or nearby columns of rows 42-47 to confirm which row is min, max, median, mean, 25th, 75th. Match the formula to the label.

## 4 – Weighted mean in H50:L50

For each column `c`:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
(Adjust column letter for each of H-L.)

## 5 – Save

Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets.

## 6 – Verification

Reopen `/root/output/result.xlsx` with openpyxl (data_only=False) and print:
- Cells H12, I12, L17 to confirm lookup formulas are present.
- Cells H35, L40 to confirm net balance formulas.
- Cells H42, H47 to confirm stats formulas.
- Cell H50 to confirm weighted mean formula.

Also check that no new sheets were added and that the workbook has exactly the original sheets.

If any cell is None or empty, debug immediately by re-reading the Data sheet layout and adjusting MATCH ranges.

## Key Warnings
- The failed hospital-bedflow task produced None values because data ranges were wrong. Triple-check that your INDEX/MATCH ranges actually cover the data on the Data sheet.
- After inspection, if the Data sheet has years in rows (not columns), you may need to swap the INDEX/MATCH orientation. Adapt accordingly.
- Use `$` for absolute references correctly: series code column must be column-absolute, year row must be row-absolute.
- Write formulas as strings (e.g., `cell.value = '=INDEX(...)'`), not computed values.

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