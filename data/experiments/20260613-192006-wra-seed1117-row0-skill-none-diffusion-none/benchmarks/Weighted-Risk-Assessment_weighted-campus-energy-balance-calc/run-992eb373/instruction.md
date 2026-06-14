# Task Instruction

Execute the following steps precisely to complete the weighted campus energy balance workbook task.

## Step 0: Inspect the workbook
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Open `/root/output/result.xlsx` using openpyxl (with `data_only=False` to preserve formulas).
3. Print the sheet names to confirm `Task` and `Data` exist.
4. On sheet `Task`, print:
   - Row 10 (to see the year headers in columns H–L)
   - Column D rows 12–17, 19–24, 26–31 (to see the series codes)
   - Row 35 label and rows 35–40 column D (to see campus names/codes for Net renewable balance)
   - Rows 42–47 column D or G (to see which stat is which: min, max, median, mean, 25th, 75th percentile)
   - Row 50 label
   - Cells H26:L31 current content (to understand Baseline Energy Demand block)
5. On sheet `Data`, print rows 21–38 fully (all columns with data) to understand the lookup table structure: which row/column holds series codes, which holds years, and where the values are.
6. Print the exact column letters and row numbers of the Data table boundaries.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write formulas using one of the allowed patterns. The most robust is typically `INDEX(MATCH,MATCH)`. For each cell in these three blocks:
- The row lookup key is the series code in column D of that row on sheet `Task`.
- The column lookup key is the year in row 10 of that column on sheet `Task`.
- The data source is on sheet `Data` rows 21:38.

Concretely, if the Data table has series codes in (say) column A and years in row 20 (or row 21), adapt accordingly. A typical formula pattern would be:

`=INDEX(Data!$B$22:$Z$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$Z$21, 0))`

Adjust the exact ranges based on what you observe in step 0. The key requirements:
- Use absolute references for the data range and lookup arrays, with mixed references ($D12 for row-absolute column, H$10 for column-absolute row) so formulas can be filled across the block.
- Use `0` for exact match in MATCH.
- Every formula must use one of the allowed patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.

Set these formulas for all cells in:
- H12:L17 (6 rows × 5 columns = 30 cells)
- H19:L24 (30 cells)
- H26:L31 (30 cells)

## Step 2: Net renewable balance and statistics

For H35:L40, the formula is:
`(Renewable Generation - Grid Consumption) / Baseline Energy Demand * 100`

You need to identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to Renewable Generation, Grid Consumption, and Baseline Energy Demand. Check the labels in column D or nearby cells from Step 0 inspection. Then for each cell, e.g. H35:
`=(H12 - H19) / H26 * 100`
(Adjust row offsets based on which block is which. The six rows correspond to six campuses.)

For H42:L47 (column-wise statistics over H35:L40):
- Row 42 (min): `=MIN(H35:H40)` — but check which row is which statistic from the labels
- Row 43 (max): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`
Match each row to the correct statistic label found in the inspection.

## Step 3: Weighted mean in H50:L50

Use SUMPRODUCT with the Net renewable balance percentages (H35:H40) as values and Baseline Energy Demand (H26:H31) as weights:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

Fill for columns H through L.

## Step 4: Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Reopen it and print a sample of cells to verify formulas are present (not just values).
3. Verify no new sheets were added.
4. Verify formatting was not changed (do not modify any cell styles, colors, fonts, borders, etc.).

## Important constraints
- Use openpyxl throughout. Do NOT use xlsxwriter (it cannot modify existing files).
- When writing formulas, assign them as strings starting with `=` to the cell's `.value` property.
- Do NOT overwrite any cells outside the specified ranges.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Preserve all existing formatting by not touching cell styles.
- If the Data sheet structure is different from what's assumed above, adapt the formula ranges accordingly based on your inspection. The inspection in Step 0 is critical — do it thoroughly before writing any formulas.

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