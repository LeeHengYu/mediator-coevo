# Task Instruction

Execute the following steps precisely to complete the weighted campus energy balance workbook task.

## Pre-work: Inspect the workbook

1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx` (create `/root/output/` if needed).
2. Using `openpyxl`, open `/root/output/result.xlsx` and inspect:
   - Sheet `Task`: Print the contents of rows 1–55, columns A–L. Pay special attention to:
     - Row 10 (years in H10:L10)
     - Column D rows 12–31 (series codes)
     - The structure of rows 35–50 (labels, campus names, formulas if any)
     - Any existing formulas or values in the yellow cell ranges
   - Sheet `Data`: Print rows 1–40, focusing on row 21–38 structure. Identify how data is laid out (which row/column has series codes, which has years, where values are).
   - Print cell formats/fills for a few yellow cells to confirm they are the target cells.
3. Print your findings before proceeding.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a spreadsheet formula (not a Python-computed value) that:
- Takes two inputs: the series code from column D of that row, and the year from row 10 of that column.
- Looks up the value from sheet `Data` rows 21:38.
- Uses one of the approved patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH.

Based on your inspection of the Data sheet layout, choose the most natural lookup pattern. For example, if Data has series codes in a column and years in a row header, INDEX(MATCH,MATCH) is likely cleanest.

IMPORTANT: Use absolute references for the Data lookup range and relative/mixed references appropriately so formulas work correctly across the filled range. The formulas must be Excel formulas stored in the cells (use `cell.value = '=FORMULA...'` in openpyxl), NOT computed Python values.

Example pattern (adjust based on actual Data layout):
- If Data!A21:A38 has series codes and Data!B20:F20 (or similar) has years:
  `=INDEX(Data!$B$21:$F$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$F$20, 0))`
- Adjust all references to match the actual layout you discovered.

## Step 2: Net renewable balance formulas in H35:L40 and statistics in H42:L47

For H35:L40 (6 campus rows × 5 year columns):
- The formula is: `(Renewable Generation - Grid Consumption) / Baseline Energy Demand * 100`
- Renewable Generation values are in H12:L17, Grid Consumption in H19:L24, Baseline Energy Demand in H26:L31.
- So for cell H35: `=(H12-H19)/H26*100` (adjust row references based on which campus maps to which row — the first campus in rows 12,19,26 maps to row 35, etc.)
- Verify that the campus order is consistent across all three blocks and the Net renewable balance block.

For H42:L47 (statistics, column-wise over H35:L40):
- Row 42: `=MIN(H$35:H$40)` (or whichever row label says MIN)
- Row 43: `=MAX(H$35:H$40)`
- Row 44: `=MEDIAN(H$35:H$40)`
- Row 45: `=AVERAGE(H$35:H$40)`
- Row 46: `=PERCENTILE(H$35:H$40, 0.25)`
- Row 47: `=PERCENTILE(H$35:H$40, 0.75)`
- CHECK the actual labels in column A/B/C/D for rows 42–47 to determine which statistic goes in which row. Match the formula to the label.

## Step 3: Weighted mean in H50:L50

For each column H through L in row 50:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net renewable balance percentages using Baseline Energy Demand as weights. The task says to use SUMPRODUCT. The formula should be:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

## Final steps

1. After writing all formulas, save the workbook.
2. Re-open and verify:
   - Cells H12:L17, H19:L24, H26:L31 contain formula strings (start with '=')
   - Cells H35:L40 contain formula strings
   - Cells H42:L47 contain formula strings
   - Cells H50:L50 contain formula strings
   - No extra sheets were added
   - Print a sample of formulas from each section to confirm correctness
3. Confirm the file is saved at `/root/output/result.xlsx`.

## Critical constraints
- Do NOT use `data_only=True` when reading — you need to write formulas, not values.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting (don't modify fonts, fills, borders, etc. — only write to cell `.value`).
- All formulas must be Excel-compatible spreadsheet formulas.
- When writing formulas with openpyxl, just assign the formula string to `cell.value`.

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