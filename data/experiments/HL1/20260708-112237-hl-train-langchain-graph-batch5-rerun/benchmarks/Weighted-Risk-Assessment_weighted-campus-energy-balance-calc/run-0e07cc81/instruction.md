# Task Instruction

Execute the following steps carefully to complete the weighted campus energy balance workbook task.

## 0. Inspect the workbook

1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx` (create `/root/output/` if needed).
2. Open the workbook with openpyxl (use `data_only=False` so you can write formulas).
3. Inspect sheet `Task`:
   - Print rows 1–55 of columns A–M to understand the layout: headers, series codes in column D, years in row 10, and the yellow cell regions.
   - Specifically print the values in D12:D17, D19:D24, D26:D31, D35:D40 to see the series codes for each campus/metric block.
   - Print row 10 columns H–L to see the year headers.
4. Inspect sheet `Data`:
   - Print rows 1–5 and rows 18–40 to understand the data layout: which row has which field, which columns hold which years, where the series codes appear.
   - Identify the exact column that holds series codes and the row that holds years, so you can build correct MATCH references.
   - Print the full header row and a few data rows to confirm the structure.

## 1. Populate H12:L31 with lookup formulas

For each cell in the three blocks (H12:L17, H19:L24, H26:L31):
- The formula must look up the value from sheet `Data` rows 21–38 using:
  - The series code from column D of the same row on sheet `Task`
  - The year from row 10 of the same column on sheet `Task`
- Use an INDEX/MATCH/MATCH pattern (or VLOOKUP with MATCH, HLOOKUP with MATCH, or XLOOKUP with MATCH — pick one pattern and use it consistently).
- IMPORTANT: Before writing formulas, confirm exactly which column in sheet `Data` contains the series codes (it might be column A, B, C, etc.) and which row contains the years. Build your MATCH ranges accordingly.
- Example pattern with INDEX/MATCH/MATCH (adjust ranges based on inspection):
  `=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))`
  Adjust the column/row references to match the actual layout you discovered.
- Write the formula into each cell. Use absolute references for the data range and MATCH lookup arrays, and mixed references ($D12 for row-locked series code column, H$10 for column-locked year row) so the pattern is consistent.

## 2. Net renewable balance (H35:L40)

For each cell in H35:L40, write a formula that computes:
`(Renewable Generation - Grid Consumption) / Baseline Energy Demand * 100`

where:
- Renewable Generation values are in H12:L17 (same row offset: row 35 corresponds to row 12, row 36 to row 13, etc.)
- Grid Consumption values are in H19:L24 (same row offset: row 35 corresponds to row 19, etc.)
- Baseline Energy Demand values are in H26:L31 (same row offset: row 35 corresponds to row 26, etc.)

So for cell H35: `=(H12-H19)/H26*100`
For cell H36: `=(H13-H20)/H27*100`
...and so on through H40: `=(H17-H24)/H31*100`
Same pattern across columns H through L.

## 3. Summary statistics (H42:L47)

For each column (H through L), calculate column-wise statistics over the 6 campus values in rows 35:40:
- Row 42: MIN, e.g., `=MIN(H35:H40)`
- Row 43: MAX, e.g., `=MAX(H35:H40)`
- Row 44: MEDIAN, e.g., `=MEDIAN(H35:H40)`
- Row 45: AVERAGE (simple mean), e.g., `=AVERAGE(H35:H40)`
- Row 46: 25th percentile, e.g., `=PERCENTILE(H35:H40,0.25)`
- Row 47: 75th percentile, e.g., `=PERCENTILE(H35:H40,0.75)`

IMPORTANT: Check the labels in column A/B/C/D for rows 42–47 to confirm which row is which statistic. Map MIN/MAX/MEDIAN/MEAN/25th/75th to the correct rows based on the actual labels. Do NOT assume the order above — verify it.

## 4. Weighted mean (H50:L50)

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net renewable balance percentages (H35:H40) weighted by Baseline Energy Demand (H26:L31). Use SUMPRODUCT as required by the instructions.

## 5. Save and verify

1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and print cells H12, H19, H26, H35, H42:H47, H50 to verify they contain formulas (not just values).
3. Confirm no new sheets were added.
4. Confirm the file is saved and exists at the correct path.

## Critical notes
- Do NOT use `data_only=True` when loading — you need to preserve and write formulas.
- Do NOT delete or modify any existing content outside the specified cells.
- Do NOT add sheets, macros, VBA, or external links.
- All formulas must be Excel-compatible spreadsheet formulas (strings starting with `=`).
- Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) for maximum compatibility, unless you determine the workbook already uses a specific variant.
- When writing formulas with openpyxl, just assign the formula string to the cell value, e.g., `ws['H12'] = '=INDEX(...)'`.
- Preserve all existing formatting by not touching styles.

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