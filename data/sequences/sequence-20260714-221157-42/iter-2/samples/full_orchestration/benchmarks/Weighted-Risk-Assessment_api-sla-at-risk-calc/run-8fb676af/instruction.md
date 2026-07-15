# Task Instruction

Execute the following steps in order to produce `/root/output/result.xlsx`.

## Preliminary Inspection

1. Create the output directory: `mkdir -p /root/output`
2. Open and inspect `/root/data/workbook.xlsx` using `openpyxl` (with `data_only=False` so you see formulas, not cached values).
   - Print the sheet names to confirm `Task` and `Data` exist.
   - Print the contents of `Task` sheet rows 10–50, columns D–L, so you can see:
     - Row 10: the year headers in H10:L10
     - Column D rows 12–17, 19–24, 26–31: the series codes
     - Rows 35–40: labels for the six services for Net SLA buffer
     - Rows 42–47: labels for min, max, median, mean, 25th, 75th percentile
     - Row 50: label for Platform SLA Coalition
   - Print the `Data` sheet rows 21–38 (all columns) to understand the data layout: which row holds which series, which column holds which year, and what the header row/column structure is.
   - Also print `Data` row 1 (or whichever row has headers) and column A/B of rows 21–38 to understand the lookup key column and the year row.

## Understanding the Layout

Before writing any formulas, determine:
- **Data sheet structure**: Identify the row that contains years (likely row 20 or a header row) and the column that contains series codes (likely column A or B). Note the exact cell references.
- **Three blocks on Task sheet**: H12:L17 (block 1), H19:L24 (block 2), H26:L31 (block 3) each have 6 rows × 5 columns. Each row's series code is in column D of that row. Each column's year is in row 10 of that column (H10, I10, J10, K10, L10).

## Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write an INDEX/MATCH formula:
```
=INDEX(Data!<data_range>, MATCH($D<row>, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```

Adjust the references based on what you found during inspection:
- `<data_range>`: the rectangular range on Data sheet covering rows 21:38 and the relevant value columns
- `<series_code_column>`: the column on Data sheet that holds the series codes (same rows 21:38)
- `<year_row>`: the row on Data sheet that holds the year values

Make sure:
- The column reference for D is absolute on column ($D) and relative on row so it changes per row
- The row reference for year is absolute on row ($10) and relative on column so it changes per column
- Use exact match (0) for both MATCH functions

Write these formulas using openpyxl by setting `cell.value = '=INDEX(...)'` as a string.

## Step 2: Net SLA Buffer in H35:L40 and Statistics in H42:L47

Based on the inspection, identify which block is "Latency Budget Preserved" (likely H12:L17), which is "Latency Budget Consumed" (likely H19:L24), and which is "Covered Request Capacity" (likely H26:L31). Verify this from the labels in the Task sheet (check column C or D or nearby cells for block headers like rows 11, 18, 25).

For each cell in H35:L40 (6 services × 5 years), the formula is:
```
=(<Preserved_cell> - <Consumed_cell>) / <Capacity_cell> * 100
```
where the row offset maps the same service (row 35 maps to row 12/19/26, row 36 to 13/20/27, etc.) and the column stays the same (H maps to H, etc.).

For H42:L47, write column-wise aggregate formulas over H35:H40 (adjusting column for each):
- Row 42: `=MIN(H35:H40)` (or whichever row is minimum — check the labels in column D/E/F/G for rows 42-47)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

**Important**: Check the actual labels in column D (or nearby) for rows 42–47 to match the correct statistic to the correct row. The order might differ from what I listed above.

## Step 3: Weighted Mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net SLA buffer percentages weighted by Covered Request Capacity.

## Saving

- Save the workbook to `/root/output/result.xlsx` using openpyxl.
- Do NOT change any formatting, do NOT add sheets, macros, VBA, or external links.
- After saving, re-open the file and print a sample of the formula cells (e.g., H12, H35, H42, H50) to verify the formulas were written correctly.

## Critical Reminders

- All formulas must be Excel formula strings (starting with `=`), not Python-computed values.
- Use `openpyxl.load_workbook('/root/data/workbook.xlsx')` without `data_only` (default is False) to preserve existing formulas.
- Do not alter any existing cell values outside the specified ranges.
- Double-check that the Data sheet range references in your INDEX/MATCH formulas exactly cover rows 21:38 as specified.
- When writing PERCENTILE formulas, use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) for maximum compatibility, unless inspection shows the workbook already uses a specific variant.

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