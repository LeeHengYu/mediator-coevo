# Task Instruction

Complete the following task to update an Excel workbook with formulas.

## Phase 0: Setup and Inspection

1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Install openpyxl if needed: `pip install openpyxl`.
3. Inspect the workbook structure thoroughly using openpyxl. Specifically:
   - Read sheet names to confirm `Task` and `Data` exist.
   - On sheet `Task`: read row 10 (especially H10:L10) to find the year headers. Read column D rows 12-31 to find the series codes. Read rows 35-50 to understand labels and structure. Read H35:L50 to see what's already there. Read the labels in column A or B/C for rows 12-17, 19-24, 26-31, 35-40, 42-47, 50.
   - On sheet `Data`: read rows 21-38 to understand the data layout — identify which column has the series codes, which row has year headers, and how the data is arranged (is it a vertical table with series codes in a column and years across columns, or vice versa?).
   - Print all of this information clearly before proceeding.

## Phase 1: Populate Lookup Formulas (H12:L17, H19:L24, H26:L31)

Using openpyxl, write spreadsheet formulas (not computed values) into the yellow cells. Each formula should use one of the allowed lookup patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH.

For each cell in the ranges H12:L17, H19:L24, H26:L31:
- The lookup key inputs are: (a) the series code from column D of that row, and (b) the year from row 10 of that column.
- The data source is sheet `Data` rows 21:38.
- Choose INDEX/MATCH as the most reliable pattern. The formula pattern should be something like:
  `=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))`
  Adjust the exact ranges based on what you discover in the inspection.

IMPORTANT: Use `$D12` (absolute column, relative row) for the series code reference and `H$10` (relative column, absolute row) for the year reference, so formulas can vary correctly across the grid. Adjust row/column anchoring appropriately.

IMPORTANT: When writing formulas with openpyxl, you must set the cell value to a string starting with `=`. Do NOT use `data_only` mode for writing.

## Phase 2: Net Production Slack (H35:L40)

Based on the inspection, identify which row ranges correspond to:
- `Finished Output` (should be one of the blocks H12:L17, H19:L24, H26:L31)
- `Scrap And Rework` (another block)
- `Rated Production Capacity` (another block)

Read the labels on the Task sheet (likely in rows 11, 18, 25 or nearby) to determine which block is which.

For each cell in H35:L40, write the formula:
`=(Finished_Output_cell - Scrap_And_Rework_cell) / Rated_Production_Capacity_cell * 100`

For example, if Finished Output is rows 12-17, Scrap And Rework is rows 19-24, and Rated Production Capacity is rows 26-31, then H35 would be:
`=(H12-H19)/H26*100`

Adjust based on actual inspection results.

## Phase 3: Summary Statistics (H42:L47)

For each column H through L, write formulas in rows 42-47 for: MIN, MAX, MEDIAN, AVERAGE, PERCENTILE (25th), PERCENTILE (75th) over the corresponding column in rows 35:40.

Check the labels in column A/B/C for rows 42-47 to determine the exact order. Then write:
- MIN: `=MIN(H35:H40)` (or equivalent)
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- AVERAGE: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)`

Match the order to the row labels found during inspection.

## Phase 4: Weighted Mean (H50:L50)

For each column H through L, write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the Net production slack percentages (H35:H40) as values and the Rated Production Capacity block (H26:L31) as weights. Adjust the Rated Production Capacity range reference if the inspection reveals it's a different block.

## Phase 5: Save and Validate

1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and verify:
   - All cells in H12:L17, H19:L24, H26:L31 contain formula strings (start with `=`).
   - All cells in H35:L40 contain formula strings.
   - All cells in H42:L47 contain formula strings.
   - All cells in H50:L50 contain formula strings.
   - No new sheets were added.
   - Print a sample of formulas from each range to confirm correctness.
3. Do NOT add any sheets, macros, VBA, external links, or helper tabs.

## Critical Notes
- Work only inside the existing `Task` and `Data` sheets.
- Preserve all existing formatting (do not clear or overwrite non-target cells).
- All values written must be Excel formulas (strings starting with `=`), not computed Python values.
- The inspection phase is essential — do not skip it. The exact row/column layout determines every formula.

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