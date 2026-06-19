# Task Instruction

Complete the following steps to update the workbook. Read carefully before editing.

## Phase 0: Inspect the workbook

1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Using openpyxl, open `/root/output/result.xlsx` and inspect both sheets thoroughly:
   - On sheet `Task`: print rows 1-55, focusing on columns A-L. Pay special attention to:
     - Row 10 (years)
     - Column D rows 12-31 (series codes)
     - The labels/headers for the three blocks (rows 12-17, 19-24, 26-31) to identify which block is Finished Output, Scrap And Rework, and Rated Production Capacity
     - Rows 35-50 structure and labels
     - Cell fills/colors to confirm yellow cells
   - On sheet `Data`: print rows 1-40, focusing on the structure of rows 21-38 to understand the lookup source layout (which dimension has series codes, which has years, how data is arranged).
3. Print all findings clearly before proceeding.

## Phase 1: Populate lookup formulas (H12:L17, H19:L24, H26:L31)

For each yellow cell in these three blocks, write a formula using INDEX+MATCH (or another allowed pattern: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH) that:
- Uses the series code from column D of that row
- Uses the year from row 10 of that column
- Looks up the value from sheet `Data` rows 21:38

IMPORTANT: Determine the exact data layout on `Data` sheet first. If data is arranged with series codes in one column and years across columns (or vice versa), choose the appropriate lookup pattern. Use absolute references where needed (e.g., anchor the lookup arrays). Make sure the `MATCH` for years references row 10 on the `Task` sheet with the column varying and the row fixed.

Use openpyxl to write these formulas as strings (e.g., cell.value = '=INDEX(...)'). Do NOT use data_only mode. Preserve all existing formatting - do not modify font, fill, border, alignment, or number format of any cell.

## Phase 2: Net production slack (H35:L40)

Based on the block labels identified in Phase 0, write formulas in H35:L40 that compute:
`(Finished_Output_cell - Scrap_And_Rework_cell) / Rated_Production_Capacity_cell * 100`

Each cell should reference the corresponding cell from the appropriate blocks in rows 12-31. For example, if row 12 is plant 1's Finished Output and row 19 is plant 1's Scrap And Rework and row 26 is plant 1's Rated Production Capacity, then H35 = (H12 - H19) / H26 * 100. Adjust based on actual layout.

## Phase 3: Summary statistics (H42:L47)

In rows 42-47, for each column H through L, write formulas for the column-wise statistics over H35:L40 (the 6 net production slack values). Based on the labels in column D/E/F/G for rows 42-47, place:
- MIN: =MIN(H35:H40) (adjust column)
- MAX: =MAX(H35:H40)
- MEDIAN: =MEDIAN(H35:H40)
- AVERAGE: =AVERAGE(H35:H40)
- 25th percentile: =PERCENTILE(H35:H40,0.25)
- 75th percentile: =PERCENTILE(H35:H40,0.75)

Match each formula to the correct row based on the labels found in Phase 0.

## Phase 4: Weighted mean (H50:L50)

For each column H through L, write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
(adjust column letter accordingly)

This computes the weighted mean of net production slack percentages weighted by Rated Production Capacity.

## Phase 5: Save and verify

1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the file and verify:
   - All target cells contain formula strings (not None or plain values)
   - Print a sample of formulas from each block to confirm correctness
   - Confirm no new sheets were added
   - Confirm the file is valid xlsx

## Critical constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify any existing formatting (fills, fonts, borders, number formats).
- Do NOT use data_only=True when reading.
- All formulas must be Excel formula strings starting with '='.
- When writing with openpyxl, load with keep_vba=False (default) and preserve formatting.

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