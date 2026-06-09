# Task Instruction

Complete the following task to populate formulas in an Excel workbook and save the result.

## Phase 0: Setup and Inspection

1. Run `mkdir -p /root/output`.
2. Install openpyxl if needed: `pip install openpyxl`.
3. Write and run a Python script to inspect the workbook `/root/data/workbook.xlsx`:
   - Print all sheet names.
   - For sheet `Task`: print rows 1-55 showing all columns A through M (print cell values AND formulas if any). Pay special attention to:
     - Row 10 (the year headers)
     - Column D rows 12-31 (series codes)
     - Column D or nearby columns for rows 35-50 (labels for Step 2 and Step 3)
     - Any existing content in H12:L17, H19:L24, H26:L31 (should be empty/yellow)
     - Any existing content in H35:L47 and H50:L50
   - For sheet `Data`: print rows 1-40 showing all columns to understand the data layout, especially rows 21-38. Identify column headers, series codes, and how years are arranged.
4. Print the exact cell values so we can determine:
   - Whether Data rows 21:38 are arranged with series codes in a column and years across columns (suitable for VLOOKUP/INDEX-MATCH), or years in rows and series in columns (suitable for HLOOKUP).
   - The exact series codes in Task column D.
   - The exact years in Task row 10.
   - Which columns in Data correspond to which fields.

## Phase 1: Populate Lookup Formulas (H12:L17, H19:L24, H26:L31)

After inspecting, write a Python script using openpyxl to:

1. Open `/root/data/workbook.xlsx`.
2. For each cell in the three blocks (H12:L17, H19:L24, H26:L31), insert a spreadsheet formula (not computed values) that:
   - Uses the series code from column D of that row on sheet `Task`
   - Uses the year from row 10 of that column on sheet `Task`
   - Looks up the value from sheet `Data` rows 21:38
   - Uses one of the allowed patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH

**Important**: Based on the Data layout you discover:
- If Data has series codes in a column (e.g., column A or B) and years across columns, use `INDEX(Data!$range, MATCH(D{row}, Data!$series_column, 0), MATCH(H$10, Data!$year_row, 0))` pattern (adjust references based on actual layout).
- Make sure row references for the series code column D are relative to each row, and column references for the year in row 10 use `$10` (row-absolute) and relative column.
- Make sure Data range references are absolute (with $).
- The formula must be a string starting with `=` placed into each cell.

## Phase 2: Net Production Slack (H35:L40) and Summary Stats (H42:L47)

In the same script:

1. For H35:L40, insert formulas calculating:
   `(Finished Output - Scrap And Rework) / Rated Production Capacity * 100`
   - Based on inspection, determine which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to "Finished Output", "Scrap And Rework", and "Rated Production Capacity". Look at labels near rows 11, 18, 25 or in column B/C/D.
   - Example formula pattern: `=(H12-H19)/H26*100` if block 1=Finished Output, block 2=Scrap And Rework, block 3=Rated Production Capacity. Adjust row offsets for each of the 6 plants.
   - Use cell references, not hardcoded values.

2. For H42:L47, insert column-wise summary statistic formulas over H35:L40:
   - Row 42: `=MIN(H35:H40)` (minimum)
   - Row 43: `=MAX(H35:H40)` (maximum)
   - Row 44: `=MEDIAN(H35:H40)` (median)
   - Row 45: `=AVERAGE(H35:H40)` (simple mean)
   - Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
   - Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)
   **Check the labels in column D/E/F/G for rows 42-47 to confirm which row gets which statistic. Match the label exactly.**

## Phase 3: Weighted Mean (H50:L50)

Insert a SUMPRODUCT formula for the weighted mean:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
(Adjust column letter for each column H through L)

This uses Step 2 percentages (H35:H40) as values and Rated Production Capacity (H26:H31) as weights.

## Phase 4: Save and Validate

1. Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets.
2. Write a validation script that:
   - Opens `/root/output/result.xlsx` with openpyxl
   - Checks that all cells in H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, and H50:L50 contain formula strings (starting with `=`)
   - Prints a sample of formulas from each block for verification
   - Confirms no new sheets were added
   - Confirms the formulas use the required patterns (MATCH appears in lookup formulas, SUMPRODUCT in row 50)

## Critical Reminders
- All cell contents must be Excel formulas (strings starting with `=`), NOT computed Python values.
- Do not modify any existing cell content or formatting.
- Do not add sheets, macros, VBA, or external links.
- Inspect the actual workbook layout FIRST before writing any formulas. The exact row/column references depend on the actual structure.
- Use `data_only=False` when reading to see existing formulas.
- When saving, preserve existing formatting by loading with openpyxl and just adding formulas to the target cells.

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