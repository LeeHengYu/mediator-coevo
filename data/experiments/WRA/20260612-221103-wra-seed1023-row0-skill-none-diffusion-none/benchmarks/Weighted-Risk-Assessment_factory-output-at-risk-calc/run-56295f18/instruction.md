# Task Instruction

Complete the following steps to update an Excel workbook. Read carefully and inspect the workbook structure before writing any formulas.

## Phase 0: Inspect the workbook

1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Using openpyxl, open `/root/output/result.xlsx` and inspect both sheets thoroughly:
   - On sheet `Task`: print rows 1-55 for columns A through M (print cell values AND note any merged cells). Pay special attention to:
     - Row 10 (the year headers)
     - Column D rows 12-31 (series codes)
     - Column D or nearby columns for rows 35-40 (plant names/identifiers)
     - The labels in rows 42-47 (statistical measures)
     - Row 50 label
     - Identify which columns are H through L (should be columns 8-12)
   - On sheet `Data`: print rows 1-40 for all populated columns. Pay special attention to:
     - The structure of rows 21-38 (the source data)
     - How data is organized (what are row headers, column headers, etc.)
     - Identify the exact row/column layout for lookup purposes
3. Print the exact cell coordinates and values so you understand the mapping between series codes, years, and data.

## Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on your inspection, write formulas into the yellow cells. Each formula must:
- Use the series code from column D of the SAME row on sheet `Task`
- Use the year from row 10 of the SAME column on sheet `Task`
- Look up the corresponding value from sheet `Data` rows 21:38
- Use one of these patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH

IMPORTANT: When constructing the lookup formula, ensure:
- The `Data` sheet reference range for rows is anchored to rows 21:38
- The series code column and year row on `Data` are correctly identified from your inspection
- Use absolute references (with $) where appropriate so formulas work across the block
- The MATCH for years should match against the actual year header row on the `Data` sheet
- The MATCH for series codes should match against the actual series code column on the `Data` sheet

Use INDEX(MATCH, MATCH) as the preferred pattern since it handles 2D lookups naturally. The formula pattern should be something like:
`=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))`

Adjust the exact ranges based on what you find in the inspection.

## Phase 2: Calculate Net Production Slack in H35:L40

The formula is: `(Finished Output - Scrap And Rework) / Rated Production Capacity * 100`

From your inspection, identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to:
- Finished Output
- Scrap And Rework  
- Rated Production Capacity

Then write the formula for each cell in H35:L40. For example, if H12:L17 is Finished Output, H19:L24 is Scrap And Rework, and H26:L31 is Rated Production Capacity, the formula in H35 would be: `=(H12-H19)/H26*100`. Adjust based on actual layout.

## Phase 3: Calculate statistics in H42:L47

For each column H through L, calculate column-wise statistics over the 6 values in rows 35:40:
- Row 42: MIN (e.g., `=MIN(H35:H40)`)
- Row 43: MAX (e.g., `=MAX(H35:H40)`)
- Row 44: MEDIAN (e.g., `=MEDIAN(H35:H40)`)
- Row 45: AVERAGE (e.g., `=AVERAGE(H35:H40)`)
- Row 46: PERCENTILE (25th) (e.g., `=PERCENTILE(H35:H40, 0.25)`)
- Row 47: PERCENTILE (75th) (e.g., `=PERCENTILE(H35:H40, 0.75)`)

IMPORTANT: Check the actual labels in rows 42-47 to determine the correct order of these statistics. Map each statistic to the correct row based on the label.

## Phase 4: Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the Step 2 percentages (H35:H40) as values and the Rated Production Capacity block (H26:L31) as weights. Adjust the capacity range reference if your inspection shows it's in a different block.

## Phase 5: Save and verify

1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and verify:
   - Cells H12:L17, H19:L24, H26:L31 all contain formulas (not plain values)
   - Cells H35:L40 contain formulas
   - Cells H42:L47 contain formulas
   - Cells H50:L50 contain formulas
   - Print the formula strings for a sample from each block to confirm correctness
   - No new sheets were added
   - The file is saved at the correct path

## Critical constraints
- Do NOT add any new sheets, macros, VBA, external links, or helper tabs
- Do NOT change existing formatting
- All formulas must be Excel spreadsheet formulas (not Python calculations)
- Use openpyxl to write formulas as strings (e.g., cell.value = '=INDEX(...)')
- When writing formulas with openpyxl, make sure the sheet name reference uses the correct syntax: `Data!` prefix for references to the Data sheet

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