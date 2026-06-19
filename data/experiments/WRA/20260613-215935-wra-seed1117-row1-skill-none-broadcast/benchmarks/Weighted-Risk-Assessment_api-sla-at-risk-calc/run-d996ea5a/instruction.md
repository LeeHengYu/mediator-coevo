# Task Instruction

Execute the following steps exactly to produce /root/output/result.xlsx.

## 0 – Inspect the workbook
```python
import openpyxl, os
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for s in wb.sheetnames:
    print(s)
ws_task = wb['Task']
ws_data = wb['Data']

# Print Task sheet layout: column headers, row labels, and the yellow target ranges
for r in range(1, 55):
    vals = []
    for c in range(1, 15):
        v = ws_task.cell(r, c).value
        vals.append(str(v) if v is not None else '')
    print(f"Row {r:3d}: {vals}")

print('\n--- Data sheet rows 1-45 ---')
for r in range(1, 45):
    vals = []
    for c in range(1, 20):
        v = ws_data.cell(r, c).value
        vals.append(str(v) if v is not None else '')
    print(f"Row {r:3d}: {vals}")
```
Study the output carefully before proceeding. Identify:
- The series codes in column D of rows 12-17, 19-24, 26-31 on Task sheet.
- The years in row 10 for columns H-L on Task sheet.
- The layout of Data sheet rows 21-38: which row holds which series, which column holds which year.
- The labels for the three metric blocks (Latency Budget Preserved, Latency Budget Consumed, Covered Request Capacity) and which Task rows (12-17, 19-24, 26-31) correspond to which block.
- The labels/series codes for rows 35-40 (Net SLA buffer) and rows 42-47 (statistics).
- Row 50 label (Platform SLA Coalition weighted mean).

## 1 – Write lookup formulas in H12:L17, H19:L24, H26:L31

Use INDEX/MATCH for a 2D lookup. The formula pattern for cell HX (row r, column c) should be:
```
=INDEX(Data!$B$21:$<lastcol>$38, MATCH($D<r>, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$<lastcol>$20, 0))
```
Adjust the exact Data sheet anchor references ($A$21:$A$38 for series codes, row 20 for year headers, $B$21:$<lastcol>$38 for the data block) based on what you found in Step 0. The column letter for the last data column and the row for year headers must match the actual Data sheet layout.

Use openpyxl to write the formula strings into each cell. Make sure:
- The row reference in `$D<r>` uses the actual Task sheet row of that cell.
- The column reference `H$10` uses the actual column letter of that cell (H, I, J, K, L) with an absolute row reference to the year header row.
- All Data sheet references are absolute ($).

## 2 – Net SLA buffer formulas in H35:L40

For each of the 6 services (rows 35-40) and each year column (H-L), write:
```
=(H12 - H19) / H26 * 100
```
where H12 is the Latency Budget Preserved cell, H19 is the Latency Budget Consumed cell, and H26 is the Covered Request Capacity cell for the same service and year. Adjust row numbers based on the actual mapping found in Step 0 (the i-th service in the Preserved block maps to the i-th service in the Consumed and Capacity blocks, and to the i-th row in 35-40).

## 3 – Summary statistics in H42:L47

For each year column (H-L), write formulas in rows 42-47. Check the labels in column D/E for rows 42-47 to determine the order, then use:
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- AVERAGE (simple mean): `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40, 0.25)`
- 75th percentile: `=PERCENTILE(H35:H40, 0.75)`

Match each formula to the correct row based on the label you see.

## 4 – Weighted mean in H50:L50

For each year column (H-L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net SLA buffer percentages weighted by Covered Request Capacity.

## 5 – Save
```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## 6 – Validate
Reload the saved workbook with data_only=False and spot-check that:
- Cells H12, L17, H19, L24, H26, L31 contain formula strings (start with '=').
- Cells H35, L40 contain formula strings.
- Cells H42, H47 contain formula strings.
- Cell H50 contains a formula string.
- No sheets were added or removed.
- Print all formula strings for a final review.

## Critical notes from prior failures
- The hospital-bedflow sibling task failed because cells were left empty (None). Make absolutely sure every cell in the target ranges gets a formula written to it.
- Double-check the Data sheet layout before constructing formulas. Do not assume column/row positions; read them from the inspection output.
- Use `ws_task.cell(row, col).value = '=FORMULA...'` to write formulas. openpyxl treats strings starting with '=' as formulas.
- Do NOT use data_only=True when loading for editing; use the default.
- Preserve all existing formatting by not touching any cells outside the target ranges.

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