# Task Instruction

## Task: Populate formulas and calculations in /root/data/workbook.xlsx and save to /root/output/result.xlsx

### Preparation
1. Create the output directory: `mkdir -p /root/output`
2. Install openpyxl if not already available: `pip install openpyxl`
3. Inspect the workbook structure thoroughly before writing any code:
   - Open `/root/data/workbook.xlsx` with openpyxl (data_only=False to see formulas)
   - Print the sheet names to confirm `Task` and `Data` exist
   - Print the contents of `Task` sheet rows 10-50, columns D through L, to understand:
     - Row 10: the year headers in columns H:L
     - Column D rows 12-17, 19-24, 26-31: the series codes
     - What labels are in column A or B for rows 12-17, 19-24, 26-31 (to understand which block is Finished Output, Scrap And Rework, Rated Production Capacity)
     - Rows 35-40: plant names and any existing content
     - Rows 42-47: labels for min, max, median, mean, 25th, 75th percentile
     - Row 50: Regional Output Council label
   - Print the `Data` sheet rows 21-38 structure: column headers (row 20 or 21), and a few sample rows to understand the data layout (which column has series codes, which columns/rows have years and values)
   - Pay special attention to whether Data sheet is organized with years in columns or rows

### Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas

Based on the inspection, write formulas in each yellow cell using INDEX/MATCH (or VLOOKUP with MATCH). Each formula must:
- Use the series code from column D of the current row on the Task sheet
- Use the year from row 10 of the current column on the Task sheet
- Look up the value from the Data sheet rows 21:38

IMPORTANT: Before writing formulas, determine the exact structure of the Data sheet:
- If Data has series codes in a column and years in a header row, use INDEX(MATCH for row, MATCH for column)
- If Data has series codes in a row and years in a column, adjust accordingly
- Make sure MATCH references are correct (exact match, match_type=0)

Use openpyxl to write Excel formula strings (not computed values). Example pattern:
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
Adjust the actual ranges based on what you find in the Data sheet inspection.

### Step 2: Net production slack in H35:L40 and statistics in H42:L47

First, determine which row blocks correspond to:
- Finished Output (one of the three blocks: rows 12-17, 19-24, or 26-31)
- Scrap And Rework (another block)
- Rated Production Capacity (another block)

Read the labels carefully from the Task sheet to identify which block is which.

For H35:L40, write formulas: `=(FinishedOutput - ScrapAndRework) / RatedProductionCapacity * 100`
- Each cell references the corresponding cells from the appropriate blocks
- For example, if Finished Output is rows 12-17, Scrap is 19-24, Capacity is 26-31:
  `=(H12-H19)/H26*100` for cell H35, etc.

For H42:L47, write column-wise aggregate formulas over H35:L40:
- H42: `=MIN(H35:H40)` (minimum)
- H43: `=MAX(H35:H40)` (maximum)
- H44: `=MEDIAN(H35:H40)` (median)
- H45: `=AVERAGE(H35:H40)` (simple mean)
- H46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- H47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

IMPORTANT: Check the labels in column A/B/C for rows 42-47 to confirm the exact order (min, max, median, mean, 25th, 75th). Map formulas to the correct rows based on labels.

### Step 3: Weighted mean in H50:L50

Write SUMPRODUCT formulas for the weighted mean:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

Adjust the Rated Production Capacity range (H26:H31) if it's in a different block.

### Final Steps
1. Save the workbook to `/root/output/result.xlsx`
2. Verify by reopening the saved file and printing:
   - A sample of the formulas in H12, H19, H26 to confirm lookup structure
   - The formulas in H35, H42-H47, H50 to confirm calculation structure
   - Confirm no new sheets were added
   - Confirm the file is valid xlsx

### Critical Constraints
- Do NOT add any new sheets
- Do NOT add macros, VBA, external links, or helper tabs
- Do NOT change existing formatting
- Use formula strings in cells, not computed Python values
- Preserve all existing content in the workbook

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