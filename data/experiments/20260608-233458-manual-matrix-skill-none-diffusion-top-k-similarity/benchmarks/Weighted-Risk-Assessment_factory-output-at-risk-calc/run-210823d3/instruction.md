# Task Instruction

Complete the following task to update an Excel workbook with formulas.

## Setup
1. First, copy the workbook: `cp /root/data/workbook.xlsx /root/output/result.xlsx`
2. Install openpyxl if needed: `pip install openpyxl`
3. Inspect the workbook structure thoroughly before making any changes.

## Inspection Phase
Open `/root/output/result.xlsx` with openpyxl and examine:
- Sheet `Task`: Read all cells in rows 1-55, columns A-M. Pay special attention to:
  - Row 10 (header row with years in H10:L10)
  - Column D rows 12-17, 19-24, 26-31 (series codes)
  - The labels/headers that identify what each block of rows represents (rows 12-17, 19-24, 26-31)
  - Row 35-40 labels and column D values
  - Row 42-47 labels (should be min, max, median, mean, 25th percentile, 75th percentile)
  - Row 50 label
  - Any existing formulas or values already present
- Sheet `Data`: Read all cells in rows 1-40, focusing on rows 21-38. Understand the data layout:
  - Which row/column contains series codes
  - Which row/column contains years
  - How the data is organized (is it a table with series codes in one column and years across columns, or vice versa?)

Print out ALL of this information so you can see the exact structure. Print cell values, row by row, for both sheets.

## Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas
Based on your inspection, write formulas in each yellow cell using INDEX/MATCH (or VLOOKUP with MATCH, HLOOKUP with MATCH, or XLOOKUP with MATCH). Each formula must:
- Use the series code from column D of the SAME row on sheet `Task`
- Use the year from row 10 of the SAME column on sheet `Task`
- Look up the value from sheet `Data` rows 21:38

The exact formula pattern depends on the Data sheet layout. For example, if Data has series codes in column A and years in row 20 (or similar header row), an INDEX/MATCH formula might look like:
`=INDEX(Data!$B$21:$F$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$F$20,0))`

Adjust the ranges based on what you actually find in the Data sheet. The key is:
- The row match should find the series code from column D in the appropriate column of the Data sheet
- The column match should find the year from row 10 in the appropriate header row of the Data sheet
- Use absolute references ($) appropriately so formulas can be filled across the range

## Step 2: Net Production Slack in H35:L40 and Statistics in H42:L47
First, identify which blocks correspond to:
- `Finished Output` (one of the three blocks: rows 12-17, 19-24, or 26-31)
- `Scrap And Rework` (another block)
- `Rated Production Capacity` (another block)

Read the labels carefully from the Task sheet to determine which block is which.

In H35:L40, enter formulas for each plant (6 plants, 5 years):
`= (Finished_Output_cell - Scrap_And_Rework_cell) / Rated_Production_Capacity_cell * 100`

For example, if Finished Output is rows 12-17, Scrap And Rework is rows 19-24, and Rated Production Capacity is rows 26-31, then H35 would be:
`= (H12 - H19) / H26 * 100`

Then in H42:L47, enter column-wise statistics over H35:L40:
- Row 42: `=MIN(H35:H40)` (adjust column for each)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

Match the actual row labels you find to the correct statistical function. Read the labels in column A-G for rows 42-47 to determine the correct order.

## Step 3: Weighted Mean in H50:L50
In H50:L50, calculate the weighted mean using SUMPRODUCT. The values are the Net Production Slack percentages (H35:H40 for column H), and the weights are the Rated Production Capacity values from the corresponding column in H26:L31.

Formula for H50: `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`
Repeat for columns I through L.

## Important Rules
- Use openpyxl to write formulas as strings (e.g., cell.value = '=INDEX(...)')
- Do NOT use data_only mode when writing
- Do NOT add new sheets, macros, or VBA
- Do NOT change existing formatting
- After writing all formulas, save to `/root/output/result.xlsx`
- After saving, re-open the file and verify that the formula cells contain formula strings (not None or empty)
- Print a sample of the formulas you wrote to confirm correctness

## Verification
After saving, reopen the file with openpyxl and:
1. Check that H12, L17, H19, L24, H26, L31 all contain formula strings
2. Check that H35, L40 contain formula strings
3. Check that H42, L47 contain formula strings  
4. Check that H50, L50 contain formula strings
5. Print several formulas to confirm they reference the correct cells and sheets

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