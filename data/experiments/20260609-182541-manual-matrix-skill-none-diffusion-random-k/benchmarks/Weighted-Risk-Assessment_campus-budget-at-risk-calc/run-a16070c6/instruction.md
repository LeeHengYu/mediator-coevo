# Task Instruction

Execute the following steps precisely to complete the campus budget workbook task.

## Phase 0: Inspect the workbook
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Using openpyxl, open `/root/output/result.xlsx` and inspect:
   - Sheet `Task`: print rows 1–55, focusing on columns A–L. Pay special attention to:
     • Row 10 (the year headers in H10:L10)
     • Column D rows 12–31 (the series codes)
     • Any existing values/formulas in H12:L31, H35:L47, H50:L50
     • The labels in column A or B for rows 12–17, 19–24, 26–31 (to understand which block is Committed Funding, Operating Spend, Approved Budget Base)
     • The labels for rows 35–40 (department names) and 42–47 (statistics)
   - Sheet `Data`: print rows 21–38 to understand the data layout (which row holds which series, which columns hold which years). Also print row 1 or the header row to understand column structure.
3. Print the exact cell values of H10:L10 on Task sheet (the years).
4. Print column D values for rows 12–31 on Task sheet (the series codes).
5. Print the Data sheet structure: what's in row 20 or the header row above row 21, and the first column values of rows 21–38.

## Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write formulas using `INDEX(MATCH, MATCH)` pattern. The general pattern for each cell should be:

`=INDEX(Data!<data_range>, MATCH(<series_code_ref>, Data!<series_code_column>, 0), MATCH(<year_ref>, Data!<year_header_row>, 0))`

Where:
- `<series_code_ref>` = reference to column D of the current row on the Task sheet (e.g., `$D12`)
- `<year_ref>` = reference to the year in row 10 of the current column (e.g., `H$10`)
- `<data_range>` = the rectangular data area on sheet Data corresponding to rows 21:38
- `<series_code_column>` = the column on Data sheet containing the series codes
- `<year_header_row>` = the row on Data sheet containing year headers

IMPORTANT: Adapt the exact ranges after inspecting the Data sheet layout. Use absolute row/column references (`$`) appropriately so formulas can fill across the 5 columns (H–L) and down the rows within each block. The series code reference should lock the column (`$D12`) and the year reference should lock the row (`H$10`).

Use openpyxl to write these formulas. For each cell in the three blocks (H12:L17, H19:L24, H26:L31), set the cell value to the formula string.

## Phase 2: Net budget buffer in H35:L40

Identify which row blocks correspond to:
- Committed Funding (one of the three blocks: rows 12–17, 19–24, or 26–31)
- Operating Spend (another block)
- Approved Budget Base (the third block)

From the labels on the Task sheet, determine the mapping. Then for each cell in H35:L40, write a formula:
`=(<Committed_Funding_cell> - <Operating_Spend_cell>) / <Approved_Budget_Base_cell> * 100`

For example, if Committed Funding is rows 12–17, Operating Spend is rows 19–24, and Approved Budget Base is rows 26–31, then H35 would be:
`=(H12 - H19) / H26 * 100`

Adjust row references for each of the 6 departments (rows 35–40 correspond to departments 1–6).

## Phase 3: Summary statistics in H42:L47

For each column H through L, in rows 42–47, write formulas for the six departments' net budget buffer values (H35:H40 for column H, etc.):
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`

IMPORTANT: Check the actual labels in rows 42–47 to determine the correct order of these statistics. Match the formula to the label, not to my assumed ordering.

## Phase 4: Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(<Net_budget_buffer_range> * <Approved_Budget_Base_range>) / SUM(<Approved_Budget_Base_range>)`

Or equivalently using SUMPRODUCT for the weighted mean:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

The instruction says to use SUMPRODUCT. The values are the Step 2 percentages (H35:H40) and the weights are the Approved Budget Base block (H26:H31).

## Phase 5: Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open it and verify:
   - All formula cells in H12:L17, H19:L24, H26:L31 contain formulas (not plain values)
   - All formula cells use INDEX+MATCH (or one of the allowed patterns)
   - H35:L40 contain formulas referencing the three blocks
   - H42:L47 contain statistical formulas
   - H50:L50 contain SUMPRODUCT formulas
   - No new sheets were added
   - Print a sample of formulas to confirm correctness
3. Ensure the output directory `/root/output/` exists before saving.

## Critical Notes
- Do NOT use `data_only=True` when reading—you need to preserve and write formulas.
- Do NOT delete or modify any existing content outside the specified cell ranges.
- Do NOT change formatting, add sheets, macros, VBA, or external links.
- When writing formulas with openpyxl, prefix with `=` and use Excel-style references.
- Adapt all range references based on what you actually observe in the inspection phase. My row/column assumptions may be slightly off—trust the actual file contents.
- If the statistics labels in rows 42–47 are in a different order than I assumed, match the formula to each label.

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