# Task Instruction

You must update the workbook `/root/data/workbook.xlsx` by inserting Excel formulas (not hardcoded values) into specific cells on the `Task` sheet, then save the result to `/root/output/result.xlsx`. Follow these steps exactly:

## Step 0 — Inspect the workbook structure

1. `mkdir -p /root/output`
2. Using openpyxl (with `data_only=False` so you see formulas), open `/root/data/workbook.xlsx` and inspect:
   - **Sheet `Task`**: Print rows 1–55 (all columns A–L) so you can see:
     - The series codes in column D for rows 12–17, 19–24, 26–31 (three blocks of 6 rows each).
     - The years in row 10, columns H–L.
     - The labels/layout in rows 35–50.
     - Any existing content in the yellow target cells.
   - **Sheet `Data`**: Print rows 1–40 (all columns) so you can see:
     - The header structure (which row has series codes, which row has years, how the data is laid out in rows 21–38).
     - Whether series codes are in a column and years across a row, or vice-versa.
   - Record: (a) which column on `Data` contains the series codes, (b) which row on `Data` contains the year headers, (c) the exact range of the data block (rows 21:38 and its column span), (d) the series codes from Task column D for all three blocks, (e) the years from Task row 10.

## Step 1 — Populate H12:L17, H19:L24, H26:L31 with lookup formulas

For each cell in these three blocks, write an Excel formula that:
- Uses the series code from column D of that row on `Task`.
- Uses the year from row 10 of that column on `Task`.
- Looks up the value from the `Data` sheet rows 21:38.
- Uses one of the approved patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH.

Choose INDEX/MATCH as the pattern (it's the most flexible). The formula structure should be:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column_range>, 0), MATCH(H$10, Data!<year_header_row_range>, 0))
```
Adjust the ranges based on what you found in Step 0. Use `$D12` (mixed reference, column locked) and `H$10` (mixed reference, row locked) so formulas copy correctly across the block.

Write these formulas using openpyxl by setting `cell.value = '=INDEX(...)'` as a string. Do NOT use `data_only=True` when loading.

## Step 2a — Net capacity headroom in H35:L40

For each of the 6 hospital clusters (rows 35–40) and each year column (H–L), write a formula:
```
=(H12 - H19) / H26 * 100
```
where:
- Row 12–17 block = Available Care Slots (first block)
- Row 19–24 block = Occupied Care Slots (second block)  
- Row 26–31 block = Staffed Bed Capacity (third block)

So for row 35 col H: `=(H12-H19)/H26*100`, for row 36 col H: `=(H13-H20)/H27*100`, etc. Adjust row references to match the correct cluster in each block. Verify the mapping by checking the series codes / cluster names.

## Step 2b — Summary statistics in H42:L47

For each year column (H–L), in the 6 rows 42–47, write column-wise formulas over the headroom block H35:L40:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

**Important**: Check the labels in column A/B/C/D for rows 42–47 to confirm the correct order (min, max, median, mean, 25th, 75th). Adjust the row assignments to match whatever labels are actually present.

## Step 3 — Weighted mean in H50:L50

For each year column (H–L), write:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net capacity headroom percentages, weighted by Staffed Bed Capacity.

## Step 4 — Save and verify

1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open `/root/output/result.xlsx` with openpyxl (data_only=False) and print the formula cells in the target ranges to confirm they contain formulas (not None or numbers).
3. Verify no extra sheets were added and the sheet names are unchanged.

## Critical constraints
- Do NOT use `data_only=True` when loading for editing.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting.
- All target cells must contain Excel formula strings, not Python-computed values.
- Use mixed cell references ($D12 and H$10) so formulas are correct across the block.

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