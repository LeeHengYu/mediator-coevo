# Task Instruction

You must update `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Follow these steps precisely.

## Step 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Use `openpyxl` (Python) to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: print rows 1–55, columns A–L (values AND any existing formulas). Pay special attention to:
     - Column D rows 12–17, 19–24, 26–31 (series codes for each service)
     - Row 10 columns H–L (year headers)
     - Row 35–40 (Net SLA buffer row labels / structure)
     - Row 42–47 (summary statistics labels)
     - Row 50 (Platform SLA Coalition weighted mean)
   - Sheet `Data`: print rows 21–38 fully (all columns) to understand the lookup source layout. Identify:
     - Which column holds the series code key
     - Which row holds the year headers
     - The orientation of the data (years in columns or rows)
   - Also print row 1–5 of `Data` to see any header structure.

## Step 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in the three blocks (rows 12–17, 19–24, 26–31; columns H–L):
- The formula must use TWO inputs: the series code from column D of that row, and the year from row 10 of that column.
- The lookup source is sheet `Data` rows 21:38.
- Use one of the allowed patterns: `INDEX/MATCH`, `VLOOKUP/MATCH`, `HLOOKUP/MATCH`, or `XLOOKUP/MATCH`.
- The recommended approach is `INDEX(MATCH, MATCH)` which handles 2D lookups cleanly:
  - `=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))`
  - Adjust the ranges based on what you discover in Step 0. The series code column reference must be absolute in the column ($D12), and the year row reference must be absolute in the row (H$10), so the formula copies correctly across the block.
- Use `openpyxl` to write these as string formulas (e.g., `ws['H12'] = '=INDEX(...)'`). Make sure the formulas are written as strings starting with `=`.

**Important**: Verify the exact column in `Data` that contains the series codes and the exact row that contains the year headers. The data range for INDEX should cover only the numeric data cells (not the header row or key column).

## Step 2 – Net SLA buffer (H35:L40) and summary statistics (H42:L47)

### H35:L40 – Net SLA buffer
The formula is: `(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`

Based on the three blocks:
- Rows 12–17 = first metric block (identify which is which from labels)
- Rows 19–24 = second metric block  
- Rows 26–31 = third metric block

Determine from the sheet labels which block corresponds to "Latency Budget Preserved", "Latency Budget Consumed", and "Covered Request Capacity". Then for each cell in H35:L40:
`=(Preserved_cell - Consumed_cell) / Capacity_cell * 100`

For example, if Preserved=rows 12–17, Consumed=rows 19–24, Capacity=rows 26–31:
`=(H12-H19)/H26*100` for H35, etc.

### H42:L47 – Summary statistics (column-wise over H35:L40)
Write formulas for each column (H through L):
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

**Check the labels** in column A/B/C/D for rows 42–47 to confirm which row is which statistic. Assign formulas accordingly.

## Step 3 – Weighted mean in H50:L50
For each column H–L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net SLA buffer percentages using Covered Request Capacity as weights.

## Step 4 – Save
- Save the workbook to `/root/output/result.xlsx` using openpyxl.
- Do NOT change formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.
- After saving, re-open the file and print the formula cells (H12, H19, H26, H35, H42–H47, H50) to verify they contain the expected formula strings.

## Technical notes for openpyxl
- Open with `load_workbook('/root/data/workbook.xlsx')` (do NOT use `data_only=True` since you need to preserve and write formulas).
- When writing formulas, assign a string starting with `=` to the cell's value.
- Existing formatting (fills, fonts, borders, number formats) is preserved automatically by openpyxl as long as you only set `.value` on cells.
- Use `wb.save('/root/output/result.xlsx')`.

## Validation
After saving, re-open `/root/output/result.xlsx` and:
1. Print cells H12, H19, H26 to confirm lookup formulas are present.
2. Print cells H35, H42, H45, H50 to confirm calculation formulas are present.
3. Verify no new sheets were added.
4. Verify the formulas reference the correct ranges based on your inspection.

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