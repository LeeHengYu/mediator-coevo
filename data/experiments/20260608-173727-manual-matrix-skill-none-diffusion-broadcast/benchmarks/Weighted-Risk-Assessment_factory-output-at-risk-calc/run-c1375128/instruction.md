# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

This task requires you to open an Excel workbook, inspect its structure, and populate specific cells with spreadsheet formulas. You must NOT change formatting, add sheets, or use macros/VBA.

### Step 0: Inspect the workbook structure
1. `mkdir -p /root/output`
2. Use Python with openpyxl to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Read row 10 (the year headers in columns H through L). Read column D for rows 12-17, 19-24, 26-31 to get the series codes. Read the labels/structure around rows 35-50 to understand what each block represents.
   - Sheet `Data`: Read rows 21-38 to understand the data layout — identify which column contains series codes, which row contains years, and how data is organized (is it a vertical table with series codes in one column and years across columns, or something else?).
   - Print all of this information so you can construct correct formulas.
   - Also read cells H35:H40 labels or column D rows 35-40 to understand which plants correspond to which rows.
   - Read the structure around H42:L47 to understand the stat labels (min, max, median, mean, 25th percentile, 75th percentile).
   - Read row 50 to understand the Regional Output Council weighted mean row.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a spreadsheet FORMULA (not a computed value) that:
- Uses the series code from column D of that row
- Uses the year from row 10 of that column
- Looks up the value from sheet `Data` rows 21:38
- Uses one of the allowed patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH

IMPORTANT: You must write actual Excel formulas as strings starting with '=' into the cells using openpyxl. The formulas must reference the `Data` sheet appropriately (e.g., `Data!A21:A38` or similar). Use absolute/mixed references as needed.

Before writing formulas, carefully determine:
- The exact column in Data sheet that contains the series codes (likely column A or B)
- The exact row in Data sheet that contains the year headers
- Whether the data is arranged so VLOOKUP, HLOOKUP, INDEX/MATCH etc. would work
- The correct range references

A good pattern would be INDEX/MATCH/MATCH for a two-dimensional lookup:
`=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))`

Adjust the exact ranges based on what you find in the Data sheet. Make sure:
- The row reference for the series code ($D12) uses a dollar sign on the column ($D) so it stays fixed when copied across columns
- The column reference for the year (H$10) uses a dollar sign on the row ($10) so it stays fixed when copied down rows
- The INDEX range, series code range, and year range are all consistent and correctly aligned

### Step 2: Net production slack formulas in H35:L40 and statistics in H42:L47

For H35:L40, the formula for each cell is:
`= (Finished_Output - Scrap_And_Rework) / Rated_Production_Capacity * 100`

You need to determine which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to:
- Finished Output
- Scrap And Rework  
- Rated Production Capacity

Look at labels in the Task sheet (likely in column A, B, C, or nearby cells around rows 11, 18, 25) to identify which block is which. Then for each cell in H35:L40, write a formula referencing the corresponding cells from those blocks. For example, if block 1 is Finished Output (rows 12-17), block 2 is Scrap and Rework (rows 19-24), block 3 is Rated Production Capacity (rows 26-31), then:
`H35 = (H12 - H19) / H26 * 100`
`H36 = (H13 - H20) / H27 * 100`
etc.

For H42:L47, write column-wise statistical formulas over H35:L40:
- Minimum: `=MIN(H35:H40)` (or `=MIN(H$35:H$40)`)
- Maximum: `=MAX(H35:H40)`
- Median: `=MEDIAN(H35:H40)`
- Simple mean: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- 75th percentile: `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`

Check the labels in column D (or nearby) for rows 42-47 to determine the exact order of these statistics.

### Step 3: Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net production slack percentages (H35:H40) weighted by Rated Production Capacity (H26:L31).

Wait — the instruction says to use SUMPRODUCT. The weighted mean formula using SUMPRODUCT is:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

Verify that H26:L31 is indeed the Rated Production Capacity block based on your inspection.

### Step 4: Save
Save the workbook to `/root/output/result.xlsx` using openpyxl. Make sure to NOT change any formatting.

When using openpyxl:
- Open with `load_workbook('/root/data/workbook.xlsx')` (do NOT use data_only=True, as you want to preserve existing formulas)
- Write formula strings to cells (e.g., `ws['H12'] = '=INDEX(...)'`)
- Save to `/root/output/result.xlsx`

### Step 5: Verify
After saving, reopen the file and print the formula content of a sample of cells (e.g., H12, L17, H35, H42, H50) to confirm formulas were written correctly. Also verify no extra sheets were added.

### Critical Notes:
- Formulas must be Excel formula strings, not Python-computed values
- Do NOT use data_only mode when loading
- Inspect the Data sheet thoroughly before constructing lookup formulas — get the exact layout right
- The series codes in column D and years in row 10 are the lookup keys
- Preserve all existing formatting — do not modify cell styles, fills, fonts, etc.
- If the Data sheet has a specific structure (e.g., series codes in column A, years in row 20), adapt your INDEX/MATCH ranges accordingly

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