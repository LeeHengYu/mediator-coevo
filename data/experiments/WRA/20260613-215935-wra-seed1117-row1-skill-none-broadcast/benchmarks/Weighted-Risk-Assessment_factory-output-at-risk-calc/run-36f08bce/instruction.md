# Task Instruction

You need to update the Excel workbook at `/root/data/workbook.xlsx` by populating specific cell ranges with spreadsheet formulas, then save the result to `/root/output/result.xlsx`.

## Preliminary Steps

1. Create the output directory: `mkdir -p /root/output`
2. Copy the workbook: `cp /root/data/workbook.xlsx /root/output/result.xlsx`
3. Inspect the workbook structure using openpyxl to understand:
   - On sheet `Task`: What are the series codes in column D for rows 12-17, 19-24, 26-31? What are the years in row 10 for columns H-L? What labels are in the relevant areas (H35:L40, H42:L47, H50:L50)?
   - On sheet `Data`: What is the layout of rows 21-38? Identify where series codes appear (likely in a column) and where years appear (likely in a row). Determine the exact row and column references for the data block.

Print all of this information before writing any formulas.

## Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write an INDEX/MATCH formula that:
- Uses the series code from column D of the current row
- Uses the year from row 10 of the current column
- Looks up the value from the `Data` sheet rows 21:38

The exact formula pattern depends on the Data sheet layout. If the Data sheet has series codes in one column and years in a header row, use a 2D INDEX/MATCH like:
`=INDEX(Data!<data_range>, MATCH(D12, Data!<series_code_column>, 0), MATCH(H10, Data!<year_row>, 0))`

Adjust the ranges based on what you discover in the inspection step. Make sure:
- The series code column reference and year row reference are correct
- The data range aligns with both the series code column and year row
- Use absolute references (with $) where appropriate so formulas can be filled across the block
- Row references for D column should NOT be absolute (they change per row)
- Column references for row 10 should NOT be absolute (they change per column)

Write formulas to all 90 cells (3 blocks × 6 rows × 5 columns).

## Step 2: Net Production Slack in H35:L40

Identify which block is "Finished Output" (likely H12:L17), which is "Scrap And Rework" (likely H19:L24), and which is "Rated Production Capacity" (likely H26:L31) by checking labels in column D or nearby cells on the Task sheet.

For each cell in H35:L40, write a formula:
`=(Finished_Output_cell - Scrap_And_Rework_cell) / Rated_Production_Capacity_cell * 100`

For example, H35 might be: `=(H12-H19)/H26*100`

## Step 2 continued: Summary Statistics in H42:L47

For each column H through L, in rows 42-47, write these formulas referencing the H35:L40 block:
- Row 42: `=MIN(H35:H40)` (column-wise minimum)
- Row 43: `=MAX(H35:H40)` (column-wise maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

IMPORTANT: Check the labels in column D (or nearby) for rows 42-47 to confirm the correct order of MIN, MAX, MEDIAN, AVERAGE, 25th percentile, 75th percentile. Map each statistic to the correct row based on the actual labels.

## Step 3: Weighted Mean in H50:L50

For each column H through L, write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net Production Slack percentages using Rated Production Capacity as weights.

## Final Steps

1. Save the workbook using openpyxl. Do NOT use `data_only=True` when loading.
2. Verify by reopening the saved file and checking that cells in the target ranges contain formulas (not None).
3. Specifically print the formula content of a few sample cells (e.g., H12, H35, H42, H50) to confirm they are properly set.

## Critical Notes
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify existing formatting.
- Use `openpyxl` for all Excel operations.
- When writing formulas, prefix them with `=` and write them as strings to the cell's value.
- Make sure the `Data` sheet name in formulas matches exactly (case-sensitive in the reference).

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