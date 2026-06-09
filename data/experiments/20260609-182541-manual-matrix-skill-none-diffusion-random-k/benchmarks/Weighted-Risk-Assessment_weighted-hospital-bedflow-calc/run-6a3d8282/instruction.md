# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
- Open `/root/data/workbook.xlsx` with openpyxl (keep formulas as-is: `data_only=False`).
- Print sheet names to confirm `Task` and `Data` exist.
- Print the `Task` sheet contents for rows 1-55, columns A-M, so you can see:
  - Row 10 (the year headers in H10:L10)
  - Column D rows 12-31 (the series codes)
  - The structure of H35:L40, H42:L47, H50:L50
  - Any existing content or formatting
- Print the `Data` sheet rows 1-40, focusing on the structure (headers, row layout, column layout) so you understand how the lookup source is organized (rows 21:38).
- Pay special attention to whether Data rows 21:38 are organized with series codes in a column (for VLOOKUP) or in a row (for HLOOKUP), and where the year values appear.

## 2. Understand the data layout
Based on inspection, determine:
- Which column on `Data` sheet contains the series codes
- Which row on `Data` sheet contains the years
- The exact range for the lookup table
- Whether VLOOKUP or INDEX/MATCH is most appropriate given the layout

## 3. Populate formulas in H12:L17, H19:L24, H26:L31 (Step 1)

Using openpyxl, write formulas into each yellow cell. For each cell at row `r`, column `c` (where H=8, I=9, J=10, K=11, L=12):

- The series code reference is `$D{r}` (column D of the current row)
- The year reference is `{col_letter}$10` (the year in row 10 for that column)

Use an INDEX/MATCH/MATCH pattern referencing the Data sheet. The exact formula depends on the Data layout you discovered. A typical pattern would be:

`=INDEX(Data!$B$22:$Z$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$Z$21, 0))`

Adjust the ranges based on your actual inspection of the Data sheet. The key requirements:
- Two inputs: series code from column D, year from row 10
- Source is Data rows 21:38
- Must use one of: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH

Write the formula for every cell in the three blocks (6 rows × 5 columns = 30 cells per block, 90 cells total). Use appropriate absolute/relative references so each cell correctly looks up its own series code and year.

## 4. Calculate Net Patient Flow in H35:L40 (Step 2a)

Based on the Task sheet layout, identify which rows in H12:L17 correspond to Patient Admissions, which rows in H19:L24 correspond to Patient Discharges, and which rows in H26:L31 correspond to Effective Bed Capacity. The six hospitals should be in the same order across all three blocks.

For each cell in H35:L40, write a formula:
`=(H12-H19)/H26*100`
(adjusting row references for each hospital and column for each year)

Specifically, if the first block is Admissions (rows 12-17), second is Discharges (rows 19-24), third is Bed Capacity (rows 26-31), then for row 35 col H:
`=(H12-H19)/H26*100`
For row 36 col H:
`=(H13-H20)/H27*100`
...and so on.

Verify by reading the row labels in column B or C to confirm which metric each block represents.

## 5. Calculate summary statistics in H42:L47 (Step 2b)

For each column (H through L), in the six rows 42-47, write formulas for:
- Row 42 (minimum): `=MIN(H35:H40)`
- Row 43 (maximum): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (simple mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

Check the row labels on the Task sheet to confirm the exact order of min/max/median/mean/25th/75th and adjust accordingly.

## 6. Calculate weighted mean in H50:L50 (Step 3)

For each column (H through L), write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This computes the weighted mean of Net Patient Flow using Effective Bed Capacity as weights.

## 7. Save and verify
- Save the workbook to `/root/output/result.xlsx`
- Reopen the saved file and print the formula cells to verify they contain the expected formulas
- Confirm no new sheets were added
- Confirm the file is valid xlsx

## Important constraints
- Do NOT use `data_only=True` when loading - preserve all existing formulas
- Do NOT modify any existing cell content, formatting, or structure outside the specified ranges
- Do NOT add sheets, macros, VBA, external links, or helper tabs
- If the workbook has defined styles or formatting, preserve them (openpyxl should do this by default)
- Use string formulas (e.g., `cell.value = '=INDEX(...)'`) not computed values

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