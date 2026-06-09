# Task Instruction

Complete the following task to update an Excel workbook with formulas.

## Phase 0: Setup and Inspection

1. `mkdir -p /root/output`
2. `cp /root/data/workbook.xlsx /root/output/result.xlsx` (work on the copy)
3. Use `openpyxl` to inspect the workbook thoroughly before writing any formulas:
   - Read sheet names to confirm `Task` and `Data` exist.
   - On sheet `Task`: print all content in rows 1-55, columns A-M. Pay special attention to:
     - Column D rows 12-31 (series codes)
     - Row 10 columns H-L (years)
     - Row 11 or nearby header rows that label the three blocks (H12:L17, H19:L24, H26:L31) — these likely correspond to 'Finished Output', 'Scrap And Rework', and 'Rated Production Capacity'
     - Rows 33-50 area for structure of Step 2 and Step 3 sections
     - Any labels in rows 42-47 (min, max, median, mean, 25th/75th percentile)
   - On sheet `Data`: print rows 18-40, all populated columns, to understand the lookup table structure (headers, series codes, year columns, data layout).
   - Print cell fill colors for a few yellow target cells to confirm which cells need formulas.

## Phase 1: Lookup Formulas (H12:L17, H19:L24, H26:L31)

For each cell in these three 6×5 blocks, write a spreadsheet **formula string** (not a computed value). The formula must:
- Use the series code from column D of that row and the year from row 10 of that column
- Look up the value from sheet `Data` rows 21:38
- Use one of the allowed patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH

Based on your inspection of the Data sheet structure, choose the most natural lookup pattern. For example, if Data has series codes in a column and years across a header row, INDEX(MATCH,MATCH) is natural. Make sure references are properly anchored (use $ where needed so the formula can vary by row and column correctly).

IMPORTANT: When writing formulas with openpyxl, assign a string starting with `=` to the cell value. Use `data_only=False` when loading. Make sure sheet references use the exact sheet name (e.g., `Data!` prefix).

## Phase 2: Net Production Slack (H35:L40)

Write formulas in H35:L40 that compute:
`(Finished_Output_cell - Scrap_And_Rework_cell) / Rated_Production_Capacity_cell * 100`

The three blocks from Step 1 are (based on inspection, confirm which block is which):
- H12:L17 = one metric (likely Finished Output)
- H19:L24 = another metric (likely Scrap And Rework)  
- H26:L31 = another metric (likely Rated Production Capacity)

Verify by reading labels near rows 11, 18, 25 on the Task sheet. Each row in 35-40 corresponds to the same plant/row offset as rows 12-17 (or similar — confirm by checking labels in column A-G for rows 35-40 vs 12-17).

Then in H42:L47, write column-wise summary formulas over H35:L40:
- Row 42: MIN(H35:H40) etc.
- Row 43: MAX(H35:H40) etc.
- Row 44: MEDIAN(H35:H40) etc.
- Row 45: AVERAGE(H35:H40) etc.
- Row 46: PERCENTILE(H35:H40,0.25) etc.
- Row 47: PERCENTILE(H35:H40,0.75) etc.

Confirm which row is which statistic by reading labels in column A-G for rows 42-47.

## Phase 3: Weighted Mean (H50:L50)

For each column H through L, write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net Production Slack percentages weighted by Rated Production Capacity. Adjust the cell references based on your confirmed block assignments.

## Phase 4: Save and Validate

1. Save the workbook (keep formatting intact — do NOT change fonts, fills, borders, column widths, etc.).
2. Re-open the saved file and verify:
   - All formula cells contain formula strings (start with `=`), not None or numeric values
   - Spot-check a few formulas for correctness
   - Confirm no extra sheets were added
   - Confirm the file is at `/root/output/result.xlsx`
3. Print a summary of all formulas written.

## Critical Notes
- Load the workbook with `data_only=False` to preserve existing formulas.
- Do NOT use `keep_vba=True` unless the file is .xlsm.
- Do NOT modify any cells outside the specified ranges.
- Do NOT change cell formatting (fills, fonts, borders, number formats).
- The yellow cells are the ONLY cells you should write to.
- If the Data sheet has a different layout than expected, adapt the lookup formula accordingly but stick to allowed patterns.
- Use absolute references (with $) appropriately so formulas are correct for each cell position.

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