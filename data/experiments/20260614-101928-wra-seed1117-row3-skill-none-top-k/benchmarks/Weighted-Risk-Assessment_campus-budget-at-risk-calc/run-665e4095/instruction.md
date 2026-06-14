# Task Instruction

Complete the following steps to update the workbook at `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`.

## Preliminary Investigation

1. Create the output directory: `mkdir -p /root/output`
2. Use `openpyxl` (Python) to inspect the workbook structure. Specifically:
   - List all sheet names to confirm `Task` and `Data` exist.
   - On sheet `Task`: print the contents of rows 10-50, columns A through L (at minimum). Pay special attention to:
     - Row 10 (years in H10:L10)
     - Column D rows 12-17, 19-24, 26-31 (series codes)
     - Row labels in column A or B for rows 12-17, 19-24, 26-31 to understand which block is which (Committed Funding, Operating Spend, Approved Budget Base)
     - Rows 35-40 (department names/labels for Net budget buffer)
     - Rows 42-47 (labels: min, max, median, mean, 25th percentile, 75th percentile)
     - Row 50 (Campus Budget Council weighted mean)
     - Check cell fills/colors in H12:L17 to confirm yellow cells
   - On sheet `Data`: print rows 21-38 fully (all columns with data) to understand the data layout — column headers, row headers, how series codes and years are arranged.
3. Print all findings before making any edits.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on your inspection of the `Data` sheet layout (rows 21:38), determine the correct lookup approach. The formulas must use two inputs:
- The series code from column D of the current row on `Task`
- The year from row 10 on `Task`

Use one of these patterns: `VLOOKUP`+`MATCH`, `HLOOKUP`+`MATCH`, `XLOOKUP`+`MATCH`, or `INDEX`+`MATCH`.

For each cell in the three blocks (H12:L17, H19:L24, H26:L31), write the appropriate Excel formula string. The formula should reference the `Data` sheet for the lookup range (rows 21:38). 

Key considerations:
- Determine whether the Data sheet has series codes in a column (suggesting VLOOKUP or INDEX/MATCH with vertical lookup) or in a row (suggesting HLOOKUP).
- Determine whether years are in a row header or column header on the Data sheet.
- Use appropriate absolute references ($) for the lookup range and lookup arrays so formulas can be filled across columns and down rows correctly. The series code reference should lock the column (e.g., $D12) and the year reference should lock the row (e.g., H$10).
- Write formulas as strings into cells using openpyxl (set `cell.value = '=FORMULA...'`).

Example pattern (adapt based on actual Data layout):
- If Data has series codes in column A (rows 21-38) and years in row 20 (columns B onward):
  `=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))`
- Adjust column/row references based on what you actually find.

## Step 2: Net budget buffer in H35:L40 and summary statistics in H42:L47

For H35:L40, each cell should contain a formula computing:
`(Committed Funding - Operating Spend) / Approved Budget Base * 100`

where:
- Committed Funding values are in H12:L17
- Operating Spend values are in H19:L24  
- Approved Budget Base values are in H26:L31

So for cell H35: `=(H12-H19)/H26*100`, H36: `=(H13-H20)/H27*100`, etc. Map each department row correctly (row 35↔rows 12,19,26; row 36↔rows 13,20,27; etc.).

For H42:L47, compute column-wise statistics over H35:L40:
- Row 42 (minimum): `=MIN(H35:H40)` (and similarly for columns I-L)
- Row 43 (maximum): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Verify the row labels in column A/B to confirm which row is which statistic. Match the formula to the label.

## Step 3: Weighted mean in H50:L50

For each column H through L, compute:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the Net budget buffer percentages (H35:H40) as values and Approved Budget Base (H26:L31) as weights.

## Final Steps

1. After writing all formulas, save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and verify:
   - Formulas exist (as formula strings, not just values) in all target cells: H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50.
   - Print a sample of formulas from each block to confirm correctness.
   - Confirm no extra sheets were added.
   - Confirm the file is saved and readable.
3. Do NOT add any macros, VBA, external links, helper tabs, or extra sheets.
4. Do NOT alter existing formatting — only set cell values (formulas).

## Important Notes
- Use `openpyxl` with `load_workbook(filename, keep_vba=False)` or similar. Do NOT use `data_only=True` when loading (that would strip formulas).
- When writing formulas, ensure they start with `=`.
- Be very careful about the exact Data sheet layout. Print it fully before writing any formulas.
- If the statistics labels (rows 42-47) are in a different order than min/max/median/mean/25th/75th, match the formula to the actual label in each row.

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