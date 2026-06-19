# Task Instruction

Execute the following multi-phase plan to populate formulas in the workbook and save the result.

## Phase 1: Inspect the workbook structure

1. Open `/root/data/workbook.xlsx` using openpyxl with `data_only=False`.
2. Print the sheet names.
3. On sheet `Task`:
   - Print rows 10-11 (columns A through L) to see the year headers and any labels.
   - Print rows 12-17 (columns A through L) to see the first lookup block and series codes in column D.
   - Print rows 19-24 (columns A through L) to see the second lookup block.
   - Print rows 26-31 (columns A through L) to see the third lookup block.
   - Print rows 33-50 (columns A through L) to see the derived calculation area, stats rows, and the MCEC weighted mean row.
4. On sheet `Data`:
   - Print rows 1-5 (columns A through Z or however far data extends) to understand the header layout.
   - Print rows 20-40 (columns A through Z) to see the data rows 21:38 referenced in the instructions. Pay special attention to:
     - Which column contains the series codes.
     - Which row contains the year headers.
     - The exact extent of the data matrix (first data column, last data column, first data row, last data row).
5. Record all findings before proceeding.

## Phase 2: Construct and write the lookup formulas (H12:L17, H19:L24, H26:L31)

Based on the inspection, write INDEX/MATCH formulas into each cell of the three blocks. The pattern for each cell should be:

```
=INDEX(Data!<data_matrix>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Adjust the ranges based on what you found in Phase 1:
- `<data_matrix>`: the rectangular block of numeric values on the Data sheet (rows 21:38, columns with data).
- `<series_code_column>`: the column on Data sheet that contains the series codes, spanning the same rows as the data matrix.
- `<year_header_row>`: the row on Data sheet that contains the year values, spanning the same columns as the data matrix.

Use `$D12` (column-locked) for the series code reference and `H$10` (row-locked) for the year reference so the formula can be applied across the grid correctly. Adjust the row number (12, 13, ... 17 for block 1; 19-24 for block 2; 26-31 for block 3) for each row.

Write these formulas using openpyxl by assigning the formula string to each cell's `.value`.

## Phase 3: Write the Net Renewable Balance formulas (H35:L40)

For each campus (6 rows) and each year (5 columns), the formula is:
```
=(H12 - H19) / H26 * 100
```
where H12 is from the Renewable Generation block (rows 12-17), H19 is from the Grid Consumption block (rows 19-24), and H26 is from the Baseline Energy Demand block (rows 26-31). Adjust row references for each campus row.

For example:
- H35 = `=(H12-H19)/H26*100`
- H36 = `=(H13-H20)/H27*100`
- etc.

Verify from Phase 1 inspection that the campus order is the same across all blocks.

## Phase 4: Write the summary statistics formulas (H42:L47)

For each column H through L:
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

Verify the exact row labels from Phase 1 to confirm which row is which statistic.

## Phase 5: Write the MCEC weighted mean formula (H50:L50)

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the Net Renewable Balance percentages as values and the Baseline Energy Demand as weights.

## Phase 6: Save and verify

1. Save the workbook to `/root/output/result.xlsx` (create the output directory if needed).
2. Reopen the saved file with `data_only=False` and spot-check several cells to confirm formulas are present (not None).
3. Print a few formula strings from each block to confirm correctness.

## Critical Notes
- Do NOT use `data_only=True` when writing; use `data_only=False` throughout.
- Do NOT add new sheets, macros, VBA, or external links.
- Do NOT modify any existing formatting.
- The exact ranges on the Data sheet MUST come from Phase 1 inspection. Do not guess.
- If the year header row or series code column differs from assumptions, adjust all formulas accordingly.
- Ensure `os.makedirs('/root/output', exist_ok=True)` before saving.

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