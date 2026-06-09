# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx` from `/root/data/workbook.xlsx`.

## Phase 0 – Inspect the workbook
1. Open `/root/data/workbook.xlsx` with `openpyxl` (data_only=False).
2. Print sheet names to confirm `Task` and `Data` exist.
3. On sheet `Task`: print rows 10-50 (columns A-L) to identify:
   - The year headers in H10:L10.
   - The series codes in column D for rows 12-17, 19-24, 26-31.
   - The row labels in column B/C for rows 35-47.
   - The structure of row 50 (Regional Output Council weighted mean).
4. On sheet `Data`: print rows 21-38 (all used columns) to identify:
   - The orientation of the lookup table (series codes in a column, years in a row, or vice-versa).
   - Which row/column holds series codes and which holds years.
   Record the exact header row and column positions.

## Phase 1 – Populate H12:L31 with INDEX/MATCH formulas
For each cell in the three blocks H12:L17, H19:L24, H26:L31:
- The formula must use the series code from column D of the same row and the year from row 10 of the same column.
- Use the pattern `=INDEX(data_range, MATCH(series_code_cell, series_code_column, 0), MATCH(year_cell, year_row, 0))` where the ranges refer to sheet `Data` rows 21:38.
- Anchor the data range and lookup vectors with `$` appropriately so the formula can be written per-cell or filled correctly.
- Use the inspection results from Phase 0 to set the exact `Data!` ranges. The data area, series-code column, and year row must match what you observed.

## Phase 2 – Net production slack (H35:L40)
For each of the six plants (rows 35-40) and each year column (H-L):
- Formula: `=(finished_output_cell - scrap_and_rework_cell) / rated_production_capacity_cell * 100`
- `finished_output_cell` is in the H12:L17 block (same relative row offset, same column).
- `scrap_and_rework_cell` is in the H19:L24 block.
- `rated_production_capacity_cell` is in the H26:L31 block.
- Verify the row mapping: row 35 corresponds to row 12, 19, 26; row 36→13,20,27; etc.

## Phase 3 – Summary statistics (H42:L47)
For each year column (H-L), write formulas in rows 42-47. Identify which row label maps to which function by reading column B/C labels from the inspection. The six statistics are:
- Minimum → `=MIN(Hxx:Hxx)` over the six slack cells in that column (rows 35-40).
- Maximum → `=MAX(...)`
- Median → `=MEDIAN(...)`
- Simple mean → `=AVERAGE(...)`
- 25th percentile → `=PERCENTILE(range, 0.25)`
- 75th percentile → `=PERCENTILE(range, 0.75)`
Match each statistic to the correct row based on the labels you observed.

## Phase 4 – Weighted mean (H50:L50)
For each year column, write:
`=SUMPRODUCT(slack_range * capacity_range) / SUM(capacity_range)`
where `slack_range` is the six Net production slack cells (rows 35-40) in that column, and `capacity_range` is the corresponding Rated Production Capacity cells (rows 26-31).
Alternatively, use the equivalent: `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)` (adjusted per column).

## Phase 5 – Save and verify
1. Ensure no new sheets were added.
2. Save to `/root/output/result.xlsx` (create `/root/output/` if needed).
3. Re-open the saved file and print the formula strings for a sample of cells in each block (e.g., H12, L17, H35, L40, H42, L47, H50, L50) to confirm they are correctly written.
4. Confirm the file exists and is non-empty.

## Important constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting (fonts, fills, borders, number formats).
- Use `openpyxl` only; do not use xlsxwriter or pandas ExcelWriter in a way that would recreate sheets.
- All formulas must be Excel formula strings (starting with `=`), not Python-computed values.

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