# Task Instruction

Execute the following steps in a single Python script using openpyxl to produce `/root/output/result.xlsx`.

## Pre-work
1. `mkdir -p /root/output`
2. Load `/root/data/workbook.xlsx` with `openpyxl.load_workbook('/root/data/workbook.xlsx')`. Use `data_only=False` so formulas are preserved.
3. **Inspect the layout** before writing anything:
   - Print rows 1–50 of sheet `Task`, columns A–M, showing cell values, to understand headers, series codes in column D, years in row 10, and the yellow target ranges.
   - Print rows 1–40 of sheet `Data`, columns A–Z (or however wide it goes), to understand the data table structure in rows 21–38.
   - Identify: (a) which column in `Data` holds the series codes (the lookup key), (b) which row in `Data` holds the year headers, (c) the data range for MATCH/INDEX.

## Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For each target block and each cell (row, col) in that block:
- The series code is in column D of the same row on `Task`.
- The year is in row 10 of the same column on `Task`.
- Write a formula using `INDEX/MATCH` (two-dimensional lookup) referencing the `Data` sheet rows 21:38. The formula pattern should be:
  ```
  =INDEX(Data!$B$21:$Z$38, MATCH(D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
  ```
  **But adjust the exact column/row references based on what you find in the inspection step.** The key is:
  - Row match: match the series code (from Task column D) against the series code column in Data rows 21–38.
  - Column match: match the year (from Task row 10) against the year header row in Data.
  - INDEX into the data body.
- Write the formula as a string into each cell. Do NOT compute values—write actual Excel formulas.

## Step 2: Net reliability gap in H35:L40
For each of the 6 region rows (rows 35–40) and 5 year columns (H–L):
- Identify which rows in the Task sheet hold `Successful API Requests`, `Failed API Requests`, and `Compute Capacity` for each region. Based on the block structure:
  - Block 1 (H12:L17): likely one indicator (e.g., Successful API Requests)
  - Block 2 (H19:L24): likely another indicator (e.g., Failed API Requests)
  - Block 3 (H26:L31): likely Compute Capacity
  - **Verify this from the inspection output.** The row order of regions should be the same across all three blocks and in rows 35–40.
- Write a formula: `=(H12-H19)/H26*100` adjusting row references so each region's row in block 1, block 2, block 3 maps to the corresponding row in 35–40. For example, if region order is the same, row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.

## Step 3: Summary statistics in H42:L47
For each year column (H–L), write formulas:
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`
**Check the labels in column D/E of rows 42–47 to confirm which row is which statistic. Adjust mapping accordingly.**

## Step 4: Weighted mean in H50:L50
For each year column (H–L), write:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
This computes the weighted mean of the net reliability gap percentages using Compute Capacity as weights.

## Final
- Save to `/root/output/result.xlsx` using `wb.save('/root/output/result.xlsx')`.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change any existing formatting, values, or structure.
- After saving, re-open the file and print the target cell ranges (H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50) to confirm all cells contain formula strings (not None).

## Critical Notes
- The failed run artifact shows cells were `None` because formulas weren't written correctly. Make sure every target cell gets a formula string.
- Adjust all references based on actual inspection—do not blindly use the example references above if the data layout differs.
- The inspection step is mandatory; print the layout before writing any formulas.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=easy, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.