# Task Instruction

Execute the following steps precisely to complete the weighted campus energy balance workbook task.

## Phase 0: Inspect the workbook
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Using `openpyxl`, open `/root/output/result.xlsx` and inspect:
   - Sheet `Task`: print rows 1-55 for columns A-L (values AND formulas). Pay special attention to:
     - Column D rows 12-17, 19-24, 26-31 (series codes)
     - Row 10 columns H-L (years)
     - Any existing content in H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50
     - Row labels in rows 35-40 (campus names), rows 42-47 (stat names), row 50
   - Sheet `Data`: print rows 1-40 for all columns to understand the data layout. Specifically note:
     - Row 21-38: what is in each column? Which column has the series code? Which row has years? What is the data range?
     - Identify the exact column letter that contains series codes and the row that contains year headers in the Data sheet.
3. Print all findings before writing any formulas.

## Phase 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas
Based on the inspection, write formulas into every cell in these three blocks. Each formula must:
- Use an INDEX/MATCH/MATCH pattern (or VLOOKUP with MATCH, HLOOKUP with MATCH, XLOOKUP with MATCH — pick whichever fits the Data layout best).
- Reference the series code from column D of the same row on the Task sheet.
- Reference the year from row 10 of the same column on the Task sheet.
- Look up the value from the Data sheet rows 21:38.
- Use appropriate absolute/relative references so each cell correctly picks its own series code and year.

IMPORTANT: When writing formulas with openpyxl, assign formula strings (starting with `=`) to cells. Do NOT use `data_only=True` when loading. Make sure the formulas use the correct sheet reference syntax for the Data sheet (e.g., `Data!` prefix).

Example pattern (adjust based on actual layout discovered in Phase 0):
- If Data sheet has series codes in column A rows 21-38 and years in row 1 columns B onward:
  `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$1:$Z$1, 0))`
- Adapt the exact ranges based on what you find in the Data sheet.

## Phase 2: Net renewable balance in H35:L40
For each of the six campuses (rows 35-40) and each year (columns H-L), write a formula:
`= (RenewableGeneration - GridConsumption) / BaselineEnergyDemand * 100`

where:
- RenewableGeneration = corresponding cell from the H12:L17 block (or whichever block contains Renewable Generation based on the row labels you found)
- GridConsumption = corresponding cell from the H19:L24 block (or whichever block contains Grid Consumption)
- BaselineEnergyDemand = corresponding cell from the H26:L31 block (or whichever block contains Baseline Energy Demand)

Map the blocks to the correct data categories based on the labels you find in the Task sheet. The campus order in rows 35-40 must match the campus order in the source blocks.

## Phase 3: Summary statistics in H42:L47
For each year column (H-L), write column-wise formulas:
- Row 42 (Minimum): `=MIN(H35:H40)` (adjust column)
- Row 43 (Maximum): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`

Match the stat labels found in column D/A of those rows to the correct function. Print the labels first.

## Phase 4: Weighted mean in H50:L50
For each year column, write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the net renewable balance percentages (H35:H40) weighted by baseline energy demand (H26:L31).

IMPORTANT: The task says to use SUMPRODUCT. The formula should be:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
(adjust column letters for each column H through L)

## Phase 5: Save and validate
1. Save the workbook (keep formatting, do not change sheet names, do not add sheets).
2. Re-open the saved file and verify:
   - All formula cells in H12:L17, H19:L24, H26:L31 contain formula strings starting with `=`
   - All formula cells in H35:L40 contain formula strings
   - All formula cells in H42:L47 contain formula strings
   - H50:L50 contain SUMPRODUCT formulas
   - No extra sheets were added
   - The file is saved at `/root/output/result.xlsx`
3. Print a summary of formulas written.

## Critical constraints
- Do NOT use `data_only=True` when loading the workbook for editing.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting (use `openpyxl` and only assign to `.value` for formula cells).
- Ensure `/root/output/` directory exists before saving.
- If any cell labels or layout differ from assumptions, adapt accordingly — the Phase 0 inspection is essential.

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