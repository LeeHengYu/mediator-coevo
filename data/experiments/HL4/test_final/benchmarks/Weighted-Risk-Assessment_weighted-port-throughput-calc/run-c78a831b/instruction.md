# Task Instruction

Execute the following steps in a single Python script to produce `/root/output/result.xlsx`.

## Pre-work: Inspect the workbook structure

1. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False).
2. Print the sheet names.
3. On sheet `Task`, print:
   - The contents of row 10 (columns A through L) to see the year headers.
   - The contents of column D for rows 12–31 to see the series codes.
   - The contents of rows 35–40 column D (port names for Net container flow).
   - The contents of rows 42–47 column D (stat labels: min, max, median, mean, 25th, 75th).
   - The contents of row 50 column D (CPA label).
   - The contents of H12 through L12 to check if any formulas already exist.
4. On sheet `Data`, print:
   - Row 21 through row 38, columns A through the last used column, to see the full data layout (headers, series codes, years, values).

Print all of this clearly before proceeding.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection above, write `INDEX/MATCH/MATCH` formulas into the yellow cells. The pattern for each cell should be:

```
=INDEX(Data!<value_range>, MATCH($D{row}, Data!<series_code_column>, 0), MATCH(Task!<year_cell>, Data!<year_header_row>, 0))
```

Use the inspection output to determine:
- The exact value range on `Data` (rows 21–38, data columns).
- The series code column on `Data` (the column containing the codes that match column D on Task).
- The year header row on `Data` (the row containing years that match row 10 on Task).

Use `$D{row}` (absolute column, relative row) for the series code reference and `H$10`, `I$10`, etc. (relative column, absolute row) for the year reference so formulas can be filled across the grid. Actually, since we are writing cell-by-cell in openpyxl, construct each formula with the correct absolute/relative references. The column letter for the year cell should match the current column (H–L), and the row for the series code should match the current row.

Write formulas for all cells in:
- H12:L17 (rows 12–17, columns H–L)
- H19:L24 (rows 19–24, columns H–L)
- H26:L31 (rows 26–31, columns H–L)

## Step 2: Net container flow formulas in H35:L40

For each port (rows 35–40) and each year column (H–L), write a formula:
```
=(H12 - H19) / H26 * 100
```
where the row numbers correspond to the matching port rows in each block:
- Row 35 uses rows 12, 19, 26
- Row 36 uses rows 13, 20, 27
- Row 37 uses rows 14, 21, 28
- Row 38 uses rows 15, 22, 29
- Row 39 uses rows 16, 23, 30
- Row 40 uses rows 17, 24, 31

Then for summary statistics in H42:L47, write formulas for each column (H–L):
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (AVERAGE): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

Verify the stat labels from the inspection to ensure correct row assignment (min/max/median/mean/25th/75th).

## Step 3: Weighted mean in H50:L50

For each column (H–L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## Final steps

1. Do NOT change any formatting, do NOT add sheets.
2. Save to `/root/output/result.xlsx` (create `/root/output/` directory if needed).
3. Re-open the saved file and print a sample of the formula cells (e.g., H12, H35, H42, H50) to confirm formulas were written correctly.
4. Print 'DONE' when complete.

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