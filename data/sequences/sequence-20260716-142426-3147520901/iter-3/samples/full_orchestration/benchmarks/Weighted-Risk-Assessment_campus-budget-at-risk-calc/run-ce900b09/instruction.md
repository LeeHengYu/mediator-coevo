# Task Instruction

Execute the following steps in order.

## 1 – Inspect the workbook
```bash
cd /root/data
python3 - <<'PY'
import openpyxl, json
wb = openpyxl.load_workbook('workbook.xlsx', data_only=False)
for s in wb.sheetnames:
    print(f'=== Sheet: {s} ===')
    ws = wb[s]
    print(f'  Dimensions: {ws.dimensions}')
    # Print first 50 rows (up to col M) for Task sheet
    if s == 'Task':
        for r in ws.iter_rows(min_row=1, max_row=55, max_col=13, values_only=False):
            vals = [(c.coordinate, c.value) for c in r if c.value is not None]
            if vals:
                print(f'  {vals}')
    # Print rows 1-5 and 20-40 for Data sheet
    if s == 'Data':
        for r in ws.iter_rows(min_row=1, max_row=5, max_col=20, values_only=False):
            vals = [(c.coordinate, c.value) for c in r if c.value is not None]
            if vals:
                print(f'  {vals}')
        print('  ... rows 20-40:')
        for r in ws.iter_rows(min_row=20, max_row=40, max_col=20, values_only=False):
            vals = [(c.coordinate, c.value) for c in r if c.value is not None]
            if vals:
                print(f'  {vals}')
PY
```
Record the exact layout: which column has series codes on the Task sheet (column D), which row has years (row 10), and the structure of the Data sheet rows 21-38 (header row, data orientation, column layout).

## 2 – Understand the Data sheet lookup structure
From the inspection, determine:
- Whether Data rows 21-38 are arranged with series codes in a specific column and years across columns (or transposed).
- The exact column that holds the series code in the Data sheet.
- The exact row that holds the year headers in the Data sheet.
- This will dictate whether to use VLOOKUP+MATCH, HLOOKUP+MATCH, INDEX+MATCH, or XLOOKUP+MATCH.

## 3 – Write the formulas
Create a Python script that:

a) Opens `/root/data/workbook.xlsx` with `openpyxl.load_workbook('workbook.xlsx')` (NOT data_only).

b) Gets the `Task` worksheet.

c) **Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31:**
   For each cell in these three 6×5 blocks:
   - The series code is in column D of the same row.
   - The year is in the same column of row 10.
   - Write an INDEX/MATCH formula referencing the Data sheet rows 21:38.
   - Use absolute references for the Data range and MATCH ranges.
   - Example pattern (adjust based on actual Data layout):
     `=INDEX(Data!$B$21:$Z$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$Z$20,0))`
   - **Verify the formula pattern is correct** by checking a couple of cells manually against the Data sheet.

d) **Step 2 – Net budget buffer in H35:L40:**
   The three blocks are: Committed Funding (H12:L17), Operating Spend (H19:L24), Approved Budget Base (H26:L31).
   For each cell (row r, col c) in H35:L40:
   - Map to the corresponding cells in the three blocks (same relative position).
   - Formula: `=(H12-H19)/H26*100` (adjusted for actual row/col).
   
   **Summary statistics in H42:L47** (one column at a time, rows 35-40 as the range):
   - Row 42: `=MIN(H35:H40)` etc.
   - Row 43: `=MAX(H35:H40)` etc.
   - Row 44: `=MEDIAN(H35:H40)` etc.
   - Row 45: `=AVERAGE(H35:H40)` etc.
   - Row 46: `=PERCENTILE(H35:H40,0.25)` etc.
   - Row 47: `=PERCENTILE(H35:H40,0.75)` etc.
   Check what labels exist in column G rows 42-47 to confirm the order (min, max, median, mean, 25th, 75th).

e) **Step 3 – Weighted mean in H50:L50:**
   `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)` for each column H-L.

f) Save to `/root/output/result.xlsx`:
   ```python
   import os
   os.makedirs('/root/output', exist_ok=True)
   wb.save('/root/output/result.xlsx')
   ```

## 4 – Validate
After saving, re-open `/root/output/result.xlsx` with openpyxl (data_only=False) and print the values/formulas of:
- H12, I12, L17 (should be formula strings starting with '=')
- H19, H26 (lookup formulas)
- H35, L40 (net buffer formulas)
- H42:H47 (stat formulas)
- H50, L50 (weighted mean formulas)

Also open with data_only=True and print H12, H35, H50 to check if they resolve (they may show None in openpyxl since it can't evaluate, but the formula strings must be present).

Confirm no extra sheets were added and the sheet count matches the original.

## Critical Notes
- Do NOT use data_only=True when loading for editing – formulas would be lost.
- Make sure every target cell gets a string starting with '=' assigned to cell.value.
- Do not modify any cells outside the specified ranges.
- Do not add sheets, macros, or VBA.
- Carefully inspect the Data sheet layout FIRST before writing any formulas – the exact row/column references are crucial.
- If the Data sheet has years in a row and series codes in a column, INDEX+MATCH is the cleanest pattern.
- If the summary stat order (min/max/median/mean/percentile) doesn't match what I listed, adjust based on the actual labels in column G.

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