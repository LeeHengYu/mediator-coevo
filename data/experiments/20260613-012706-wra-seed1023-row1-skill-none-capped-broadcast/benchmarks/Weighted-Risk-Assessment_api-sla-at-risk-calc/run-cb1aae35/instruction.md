# Task Instruction

Execute the following steps precisely to complete the workbook task.

## 0. Inspect the workbook

```
cp /root/data/workbook.xlsx /root/output/result.xlsx
```

Open `/root/output/result.xlsx` using openpyxl (with data_only=False so formulas are preserved). Inspect:
- Sheet `Task`: Print all cell values/formulas in rows 1-55, columns A-L. Pay special attention to:
  - Column D rows 12-17, 19-24, 26-31 (series codes)
  - Row 10 columns H-L (years)
  - The yellow cells H12:L17, H19:L24, H26:L31 (should be empty or placeholder)
  - Rows 35-40 (Net SLA buffer area), rows 42-47 (stats area), row 50 (weighted mean)
  - Any labels in columns A-G for rows 35-47 and 50
- Sheet `Data`: Print all cell values in rows 1-40, all populated columns. Focus on rows 21-38 to understand the data layout (which row has which series code, which columns have which years, orientation of the data table).

Print everything clearly before making any edits.

## 1. Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, determine:
- The series codes in column D for each row (12-17, 19-24, 26-31)
- The years in row 10 for columns H-L
- The layout of the Data sheet rows 21-38 (is it a vertical table with series codes in one column and years across columns? or horizontal?)

For each yellow cell, write a formula using INDEX/MATCH (preferred) or another allowed pattern (VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH). The formula must:
- Reference the series code from column D of the current row on sheet Task
- Reference the year from row 10 of the current column on sheet Task  
- Look up the value from Data!rows 21:38

Example pattern (adjust column/row references based on actual Data layout):
- If Data has series codes in column A and years in a header row (e.g., row 21), use something like:
  `=INDEX(Data!$B$22:$XX$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$XX$21, 0))`
- Adjust the exact ranges based on what you find in the Data sheet.

Use mixed references: lock the series code column ($D12) and the year row (H$10) so formulas can be filled across rows and columns.

## 2. Calculate Net SLA buffer in H35:L40

The formula is: `(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`

Based on the inspection, determine which block is which:
- H12:L17 = one metric block (check labels)
- H19:L24 = another metric block
- H26:L31 = another metric block

Identify which block corresponds to "Latency Budget Preserved", "Latency Budget Consumed", and "Covered Request Capacity" from the row/section labels.

For each cell in H35:L40, write the formula referencing the corresponding cells. For example, if Latency Budget Preserved is in rows 12-17, Consumed is in rows 19-24, and Covered Request Capacity is in rows 26-31:
`=(H12-H19)/H26*100` in H35, etc.

Adjust based on actual layout.

## 3. Statistics in H42:L47

For each column H through L, calculate column-wise statistics over the 6 Net SLA buffer values (H35:H40 for column H, etc.):
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

Check the labels in column A/B/C for rows 42-47 to confirm the correct order (min, max, median, mean, 25th, 75th). Adjust row assignments to match the labels.

## 4. Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses Net SLA buffer percentages as values and Covered Request Capacity as weights.

## 5. Save and verify

Save the workbook to `/root/output/result.xlsx`. Then re-open it and print all formulas in the modified cells to verify they are correct. Do NOT use data_only=True (that would lose formulas). Confirm:
- All cells in H12:L17, H19:L24, H26:L31 contain lookup formulas
- All cells in H35:L40 contain the net SLA buffer formula
- All cells in H42:L47 contain the correct statistical functions
- All cells in H50:L50 contain SUMPRODUCT-based weighted mean formulas
- No new sheets were added
- Existing formatting is preserved (use openpyxl without disturbing styles)

IMPORTANT: When writing formulas with openpyxl, prefix them with '=' and use Excel-style syntax. openpyxl stores them as strings. Use `cell.value = '=FORMULA...'`.

IMPORTANT: Do NOT call `load_workbook` with `data_only=True` as that strips formulas. Use default `data_only=False`.

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