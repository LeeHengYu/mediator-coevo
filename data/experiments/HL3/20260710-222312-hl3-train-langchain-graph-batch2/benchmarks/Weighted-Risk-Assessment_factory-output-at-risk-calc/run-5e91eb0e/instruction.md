# Task Instruction

## Task: Populate formulas and calculations in /root/data/workbook.xlsx and save to /root/output/result.xlsx

### Preparation

1. Create the output directory: `mkdir -p /root/output`
2. Install openpyxl if not already available: `pip install openpyxl`
3. Inspect the workbook structure thoroughly before making any changes:
   - Open `/root/data/workbook.xlsx` with openpyxl
   - Print the sheet names to confirm `Task` and `Data` exist
   - Print the contents of `Task` sheet rows 10-50, columns D through L, to understand the layout:
     - Row 10: identify the years in columns H:L
     - Column D rows 12-17, 19-24, 26-31: identify the series codes
     - Rows 12-17, 19-24, 26-31: identify what data blocks these are (likely three different metrics)
     - Row 35-40: identify plant names and what block labels exist
     - Row 42-47: identify the stat labels (min, max, median, mean, 25th, 75th percentile)
     - Row 50: identify the "Regional Output Council" row
   - Print the `Data` sheet rows 21-38 to understand the lookup source structure (column headers, row labels, data layout)
   - Print row 20 or the header row of the Data sheet to understand column arrangement

### CRITICAL: Inspect before coding
Do NOT write formulas until you have printed and understood:
- The exact cell references for years in row 10 (are they in H10, I10, J10, K10, L10?)
- The exact series codes in column D for each block
- The exact structure of Data!rows 21:38 (is it a vertical table with series codes in one column and years across columns, or some other layout?)
- Which row/column on the Data sheet contains the series codes and which contains the year headers

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a spreadsheet formula (not a Python-computed value) using one of these patterns:
- `INDEX(MATCH, MATCH)` — typically best for 2D lookups
- `VLOOKUP` with `MATCH` for the column
- `XLOOKUP` with `MATCH`

The formula must:
- Look up the series code from column D of the current row
- Look up the year from row 10 of the current column
- Find the value in Data sheet rows 21:38

Based on your inspection of the Data sheet layout, construct the appropriate formula. For example, if Data has series codes in column A rows 21:38 and years across row 20 starting at column B, an INDEX/MATCH formula might look like:
`=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))`

Adjust the exact ranges based on what you find in the Data sheet. The key constraint is that the lookup area must be within rows 21:38 of the Data sheet.

### Step 2: Net production slack in H35:L40 and statistics in H42:L47

First identify which of the three blocks (rows 12-17, 19-24, 26-31) corresponds to:
- Finished Output
- Scrap And Rework  
- Rated Production Capacity

This should be visible from labels on the Task sheet (check column B or C or nearby cells for block headers around rows 11, 18, 25).

For each cell in H35:L40, write a formula:
`=(Finished_Output_cell - Scrap_And_Rework_cell) / Rated_Production_Capacity_cell * 100`

These should be cell-reference formulas, not hardcoded values. The six plants in rows 35-40 should correspond to the six plants in rows 12-17 (same order).

For H42:L47, write column-wise statistical formulas over H35:L40:
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- MEAN: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- 75th percentile: `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`

Match the row order to the labels you find in the Task sheet rows 42-47. Check the labels carefully!

### Step 3: Weighted mean in H50:L50

For each column (H through L), write:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the Step 2 percentages (H35:H40) as values and Rated Production Capacity (H26:H31) as weights.

### Saving

- Save the workbook to `/root/output/result.xlsx`
- Do NOT change any existing formatting, do NOT add sheets, macros, VBA, external links, or helper tabs
- When opening with openpyxl, use `load_workbook(filename, keep_vba=False)` and do NOT pass `data_only=True` (we need to preserve and write formulas)

### Validation

After saving, re-open the result file and:
1. Verify that cells H12, L17, H19, L24, H26, L31 contain formula strings (start with '=')
2. Verify that cells H35, L40 contain formula strings
3. Verify that cells H42, L47 contain formula strings
4. Verify that cells H50, L50 contain formula strings with SUMPRODUCT
5. Print a sample of the formulas to confirm they reference the correct sheets and ranges
6. Confirm the workbook still has exactly the same sheet names as the original

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