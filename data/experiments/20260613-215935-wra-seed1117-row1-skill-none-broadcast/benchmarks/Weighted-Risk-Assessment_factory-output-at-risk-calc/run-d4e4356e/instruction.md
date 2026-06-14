# Task Instruction

Complete the following task to populate formulas in an Excel workbook. Work carefully and inspect the workbook thoroughly before writing any formulas.

## Phase 0: Setup and Inspection

1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Using openpyxl (with data_only=False to see formulas), inspect the workbook thoroughly:
   a. On sheet `Task`: Print the contents of columns A-L for rows 1-55. Pay special attention to:
      - Row 10 (years)
      - Column D rows 12-17, 19-24, 26-31 (series codes)
      - Column D or labels for rows 35-40 (plant names for Net production slack)
      - Rows 42-47 labels (min, max, median, mean, 25th, 75th percentile)
      - Row 50 label
      - Any existing formulas or values already present
   b. On sheet `Data`: Print rows 1-5 and rows 18-40, all populated columns. Identify:
      - The structure of the lookup table in rows 21:38
      - Where series codes appear (which column/row)
      - Where years appear (which column/row)
      - The orientation of the data (series codes in rows vs columns, years in rows vs columns)
3. Print the exact cell values for a few key reference points to understand the data layout.

## Phase 1: Step 1 - Lookup Formulas in H12:L17, H19:L24, H26:L31

Based on inspection, write formulas into the yellow cells. Each formula must:
- Use the series code from column D of the SAME row on sheet `Task`
- Use the year from row 10 of the SAME column on sheet `Task`
- Look up the value from sheet `Data` rows 21:38
- Use one of these patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH

IMPORTANT: Determine the correct lookup pattern by inspecting the Data sheet layout:
- If Data has series codes in a column and years across columns (or vice versa), choose the appropriate lookup function.
- Use appropriate absolute/mixed references so formulas work across the 5-column (H-L) and 6-row blocks.
- Use `INDEX(MATCH, MATCH)` if the data is a 2D table, as this is the most flexible approach.

Write formulas for all three blocks: H12:L17 (first metric block), H19:L24 (second metric block), H26:L31 (third metric block).

## Phase 2: Step 2 - Net Production Slack and Statistics

For H35:L40, determine which rows contain:
- `Finished Output` (should be one of the blocks from Step 1, e.g., H12:L17)
- `Scrap And Rework` (another block from Step 1)
- `Rated Production Capacity` (the third block, H26:L31 based on Step 3's reference)

The formula for each cell in H35:L40 is:
`= (Finished_Output_cell - Scrap_And_Rework_cell) / Rated_Production_Capacity_cell * 100`

Map the six plants (rows 35-40) to the corresponding rows in each block (rows 12-17, 19-24, 26-31) by matching plant/row order.

For H42:L47, write column-wise formulas over H35:L40:
- Row 42: MIN(H35:H40) etc.
- Row 43: MAX(H35:H40) etc.
- Row 44: MEDIAN(H35:H40) etc.
- Row 45: AVERAGE(H35:H40) etc.
- Row 46: PERCENTILE(H35:H40, 0.25) etc.
- Row 47: PERCENTILE(H35:H40, 0.75) etc.

Match the label in column A/B/C to the correct statistic function. Check the exact row-to-statistic mapping from the labels.

## Phase 3: Step 3 - Weighted Mean

For H50:L50, use SUMPRODUCT for weighted mean:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
(Adjust column references for each column H through L.)

This uses the Net production slack percentages (H35:L40) as values and Rated Production Capacity (H26:L31) as weights.

## Phase 4: Validation

1. Re-open `/root/output/result.xlsx` with openpyxl (data_only=False).
2. Verify formulas exist in all required cells: spot-check H12, L17, H19, L24, H26, L31, H35, L40, H42, L47, H50, L50.
3. Verify no new sheets were added.
4. Verify the file saves without errors.
5. Optionally open with data_only=True or use LibreOffice to verify computed values are reasonable (positive numbers, percentages in plausible range).

## Critical Notes
- Do NOT modify any formatting, styles, or existing content.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- When writing formulas with openpyxl, prefix them with `=` as strings.
- Use the exact sheet name `Data` in cross-sheet references (e.g., `Data!A21:Z38`).
- Inspect before writing. The exact row/column mapping is critical and must come from actual file inspection, not assumptions.

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