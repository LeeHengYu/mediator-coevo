# Task Instruction

## Task: Weighted Hospital Bedflow Calculation

You need to update `/root/data/workbook.xlsx` with spreadsheet formulas and save the result to `/root/output/result.xlsx`.

### Phase 0: Inspect the workbook

1. Create `/root/output/` directory if it doesn't exist.
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Print the contents of rows 1-55, columns A-L. Pay special attention to:
     - Column D (series codes) in rows 12-17, 19-24, 26-31, 35-40
     - Row 10 (years in columns H-L)
     - Rows 42-47 labels (min, max, median, mean, 25th, 75th percentile)
     - Row 50 label
     - Note which cells have yellow fill (the target cells)
   - Sheet `Data`: Print rows 1-40 to understand the data layout, especially rows 21-38. Note the structure: what's in each column, how series codes and years are organized.
3. Print the exact cell values for D12:D17, D19:D24, D26:D31, D35:D40, and H10:L10 so you know the lookup keys.
4. Examine the Data sheet rows 21:38 carefully - determine whether data is arranged with series codes in rows and years in columns, or vice versa. Identify which row/column contains series codes and which contains years.

### Phase 1: Populate lookup formulas in H12:L31

For each yellow cell in the three blocks (H12:L17, H19:L24, H26:L31), write a spreadsheet formula (not a computed value) that looks up data from the `Data` sheet rows 21:38.

The formula must use TWO inputs:
- The series code from column D of the current row (e.g., `$D12`)
- The year from row 10 of the current column (e.g., `H$10`)

Use one of these patterns: `VLOOKUP`+`MATCH`, `HLOOKUP`+`MATCH`, `XLOOKUP`+`MATCH`, or `INDEX`+`MATCH`.

Choose the pattern based on the data layout you discovered in Phase 0. For example:
- If Data has series codes in a column and years in a row header, `INDEX(MATCH,MATCH)` is most natural.
- If Data is arranged for vertical lookup, `VLOOKUP` with `MATCH` for the column index works.

IMPORTANT: Use `openpyxl` to write formula strings into cells (e.g., `ws['H12'] = '=INDEX(...)'`). Do NOT compute values in Python. The formulas must be Excel formulas stored as strings.

Make sure to anchor references appropriately with `$` signs so formulas can be consistent across the block (e.g., lock the column for the series code reference with `$D12`, lock the row for the year with `H$10`).

### Phase 2: Net Patient Flow formulas in H35:L40

For each cell in H35:L40, write a formula that calculates:
`(Patient Admissions - Patient Discharges) / Effective Bed Capacity * 100`

Based on your inspection:
- Identify which block (H12:L17, H19:L24, or H26:L31) corresponds to Patient Admissions, Patient Discharges, and Effective Bed Capacity. The labels should be visible in the Task sheet.
- Each hospital in rows 35-40 corresponds to the same hospital in the same relative position in the three blocks above.
- For example, if Admissions are in rows 12-17, Discharges in rows 19-24, and Capacity in rows 26-31, then for H35: `=(H12-H19)/H26*100`

Write these as Excel formulas, not computed values.

### Phase 3: Summary statistics in H42:L47

For each column H through L, write formulas in rows 42-47 for these statistics over the Net Patient Flow values (H35:L40 for column H, etc.):
- Row 42: Minimum → `=MIN(H35:H40)` (adjust column)
- Row 43: Maximum → `=MAX(H35:H40)`
- Row 44: Median → `=MEDIAN(H35:H40)`
- Row 45: Simple Mean → `=AVERAGE(H35:H40)`
- Row 46: 25th percentile → `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- Row 47: 75th percentile → `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Check the row labels to confirm which row is which statistic. Adjust row assignments based on actual labels.

### Phase 4: Weighted mean in H50:L50

For each column H through L, write a `SUMPRODUCT` formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of Net Patient Flow percentages weighted by Effective Bed Capacity. Adjust the capacity range if it's not H26:L31 based on your Phase 0 findings.

### Phase 5: Save and Validate

1. Save the workbook to `/root/output/result.xlsx` using `openpyxl`. Make sure NOT to use `data_only=True` when opening (so formulas are preserved).
2. Re-open the saved file and verify:
   - Cells H12:L17, H19:L24, H26:L31 contain formula strings (start with `=`)
   - Cells H35:L40 contain formula strings
   - Cells H42:L47 contain formula strings
   - Cells H50:L50 contain formula strings
   - Print a sample of formulas to confirm correctness
3. Confirm no new sheets were added and existing formatting is intact.

### Critical Rules
- Do NOT use `data_only=True` when loading the workbook.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- All target cells must contain Excel formula strings, not Python-computed values.
- Formulas must reference the `Data` sheet for lookups (Phase 1) using proper cross-sheet references like `Data!A21:A38`.
- Double-check every formula for correct sheet references, range addresses, and function syntax before saving.

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