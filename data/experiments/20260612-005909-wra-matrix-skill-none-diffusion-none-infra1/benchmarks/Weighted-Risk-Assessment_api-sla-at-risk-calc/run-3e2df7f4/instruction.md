# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Preparation

```bash
mkdir -p /root/output
pip install openpyxl
```

Open and inspect the workbook:
```python
import openpyxl
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
print('Sheet names:', wb.sheetnames)
ws_task = wb['Task']
ws_data = wb['Data']

# Print Task sheet structure to understand layout
print('\n--- Task sheet rows 1-55, cols A-M ---')
for row in ws_task.iter_rows(min_row=1, max_row=55, min_col=1, max_col=13, values_only=False):
    for cell in row:
        if cell.value is not None:
            print(f'  {cell.coordinate}: {cell.value} (type={type(cell.value).__name__})')

print('\n--- Data sheet rows 1-45, cols A-Z ---')
for row in ws_data.iter_rows(min_row=1, max_row=45, min_col=1, max_col=26, values_only=False):
    for cell in row:
        if cell.value is not None:
            print(f'  {cell.coordinate}: {cell.value} (type={type(cell.value).__name__})')
```

Study the output carefully. Identify:
- What is in column D for rows 12-17, 19-24, 26-31 (series codes)
- What is in row 10 for columns H-L (years)
- What the Data sheet rows 21-38 look like (column layout, where series codes and year values are)
- What labels are in rows 12-17 (Latency Budget Preserved), 19-24 (Latency Budget Consumed), 26-31 (Covered Request Capacity)
- What rows 35-40 labels are (Net SLA buffer services)
- What rows 42-47 labels are (min, max, median, mean, 25th, 75th percentile)
- What row 50 is (Platform SLA Coalition weighted mean)

## 1. Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write an INDEX/MATCH formula that:
- Looks up the series code from column D of that row
- Matches the year from row 10 of that column
- Searches Data sheet rows 21:38

The exact formula pattern depends on the Data sheet layout. After inspecting:
- Determine which column on Data contains the series codes (let's call it the key column, likely column A or B)
- Determine which row on Data contains the year headers
- Build the formula accordingly

Typical pattern if Data has series codes in column A and years across columns starting from B with a header row:
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```

Adjust column/row references based on actual inspection. The key constraint is: use the series code from column D of the current row AND the year from row 10 of the current column. Use INDEX+MATCH (or one of the other allowed patterns).

Write these formulas using openpyxl by setting cell.value to the formula string (e.g., `ws_task['H12'] = '=INDEX(...)'`). Do NOT use data_only mode. Make sure:
- The row reference for the series code uses `$D` with the row number (absolute column, relative row within each block)
- The column reference for the year uses the column letter with `$10` (relative column, absolute row)
- The Data ranges are fully absolute with `$`

Fill all 6 rows × 5 columns = 30 cells for each of the three blocks (90 formulas total).

## 2. Net SLA buffer in H35:L40

For each cell in H35:L40, the formula is:
```
=(LatencyBudgetPreserved - LatencyBudgetConsumed) / CoveredRequestCapacity * 100
```

The six services in rows 35-40 correspond to the six services in rows 12-17 (Preserved), 19-24 (Consumed), 26-31 (Capacity). So for cell H35:
```
=(H12-H19)/H26*100
```
For H36: `=(H13-H20)/H27*100`, etc. Adjust row mapping based on actual inspection — verify that row 35 service matches row 12, row 36 matches row 13, etc. If the order differs, match by service name.

## 3. Summary statistics in H42:L47

Based on the labels in column D/A for rows 42-47, write:
- Minimum: `=MIN(H35:H40)` (or `=MIN(H$35:H$40)`)
- Maximum: `=MAX(H35:H40)`
- Median: `=MEDIAN(H35:H40)`
- Mean: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Map each row to the correct function based on the label in that row. Fill across columns H-L.

## 4. Weighted mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net SLA buffer percentages using Covered Request Capacity as weights.

## 5. Save

```python
wb.save('/root/output/result.xlsx')
```

## 6. Verification

Reload the saved file and verify:
```python
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb2['Task']
# Check that formulas exist in key cells
for cell_addr in ['H12', 'L17', 'H19', 'L24', 'H26', 'L31', 'H35', 'L40', 'H42', 'L47', 'H50', 'L50']:
    print(f'{cell_addr}: {ws[cell_addr].value}')
```

Confirm all 90 lookup cells, 30 net-buffer cells, 30 summary cells, and 5 weighted-mean cells contain formulas (not None, not raw values). Confirm no extra sheets were added. Confirm the file exists at `/root/output/result.xlsx`.

## Critical Notes
- You MUST inspect the workbook structure first before writing any formulas. The exact cell references depend on the actual layout.
- Do NOT hardcode values; use spreadsheet formulas only.
- Do NOT add sheets, macros, or VBA.
- Preserve all existing formatting (use openpyxl.load_workbook without data_only, and don't modify styles).
- If the service order in rows 35-40 differs from rows 12-17, match by name, not by position.
- Use `PERCENTILE.INC` if you're unsure — it's the standard Excel PERCENTILE equivalent.

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