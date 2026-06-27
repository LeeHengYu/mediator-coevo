# Task Instruction

You need to update `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Work only inside the existing sheets `Task` and `Data`; do not add sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

**Before writing any code, inspect the workbook thoroughly:**
1. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False).
2. Print the contents of sheet `Task` rows 10-50, columns D-L, to understand the layout: series codes in column D, years in row 10, yellow target cells, campus names, etc.
3. Print sheet `Data` rows 21-38 to understand the source data layout (column headers, row labels, series codes, and where numeric data lives).
4. Identify the exact column letters and row numbers for the Data sheet's lookup range.

**Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31:**
For each cell in these three blocks, write an INDEX/MATCH formula that:
- Uses the series code from column D of the current row (use $D with relative row, e.g., $D12)
- Uses the year from row 10 of the current column (use relative column with $10, e.g., H$10)
- Looks up data from sheet `Data` rows 21:38

The formula pattern should be:
`=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))`

Adjust the exact range references (column letters, row numbers) based on your inspection of the Data sheet. The key is:
- The row match searches the series code column of the Data sheet
- The column match searches the header row of the Data sheet (the row containing years)
- The INDEX range covers the data area (excluding the header row and label column)

Make sure the MATCH ranges and INDEX range are consistent (same number of rows/columns). Use absolute references ($) for the Data ranges and for the $D column and $10 row anchors so formulas replicate correctly across the block.

**Step 2 – Net Renewable Balance in H35:L40:**
Based on inspection, identify which rows in the three blocks correspond to:
- Renewable Generation (likely H12:L17)
- Grid Consumption (likely H19:L24)  
- Baseline Energy Demand (likely H26:L31)

Verify this by checking the block labels on the Task sheet. Then for each cell in H35:L40:
`=(H12-H19)/H26*100` (adjusting row references for each campus row)

This calculates: (Renewable Generation - Grid Consumption) / Baseline Energy Demand * 100

**Step 2 continued – Summary statistics in H42:L47:**
For each column H through L, calculate these six statistics over the corresponding H35:L40 range:
- Row 42: `=MIN(H$35:H$40)` (column-wise MIN)
- Row 43: `=MAX(H$35:H$40)` (column-wise MAX)
- Row 44: `=MEDIAN(H$35:H$40)` (column-wise MEDIAN)
- Row 45: `=AVERAGE(H$35:H$40)` (column-wise AVERAGE)
- Row 46: `=PERCENTILE(H$35:H$40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H$35:H$40,0.75)` (75th percentile)

**CRITICAL**: Check the labels in column D (or nearby) for rows 42-47 to confirm the correct order of MIN, MAX, MEDIAN, AVERAGE, 25th percentile, 75th percentile. Match the formula to the label, not to my assumed ordering.

**CRITICAL**: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid #NAME? errors. The failed run from a similar task had #NAME? errors in the percentile rows — this was likely caused by using `PERCENTILE.INC` or `PERCENTILE.EXC` which may not be recognized. Stick to the classic function names: `MIN`, `MAX`, `MEDIAN`, `AVERAGE`, `PERCENTILE`.

**Step 3 – Weighted mean in H50:L50:**
For each column H through L:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This uses the Net Renewable Balance percentages as values and Baseline Energy Demand as weights.

**Final steps:**
1. After writing all formulas, save to `/root/output/result.xlsx` (create the output directory if needed).
2. Reopen the saved file and spot-check a few cells to confirm formulas were written (not just values).
3. Verify no #NAME? or #REF! errors by opening with data_only=True and checking that the cells don't contain error strings (note: openpyxl data_only=True may show None for formula cells that haven't been calculated by Excel, which is fine — just make sure the formula text itself is correct by also checking with data_only=False).

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