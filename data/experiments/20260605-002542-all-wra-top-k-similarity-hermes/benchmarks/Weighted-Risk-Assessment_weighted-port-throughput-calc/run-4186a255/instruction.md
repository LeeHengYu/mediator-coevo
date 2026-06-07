# Task Instruction

Execute the following steps exactly.

## 0 – Explore the workbook and test harness
```bash
cd /root
find . -name '*.py' -o -name '*.xlsx' | head -40
```
Open and inspect the test file (likely `/root/tests/test_outputs.py`) to understand:
- Which cells the verifier reads and what values it expects.
- How it loads the workbook (openpyxl data_only? or formula check?).

Then inspect the workbook:
```python
import openpyxl
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for s in wb.sheetnames:
    print(s)
ws_task = wb['Task']
ws_data = wb['Data']

# Print Task sheet layout rows 1-55, columns A-M
for row in ws_task.iter_rows(min_row=1, max_row=55, max_col=13, values_only=False):
    for cell in row:
        if cell.value is not None:
            print(f"{cell.coordinate}: {repr(cell.value)}")

# Print Data sheet rows 1-45, columns A-Z
for row in ws_data.iter_rows(min_row=1, max_row=45, max_col=26, values_only=False):
    for cell in row:
        if cell.value is not None:
            print(f"{cell.coordinate}: {repr(cell.value)}")
```
Pay close attention to:
- Column D of Task sheet (series codes in rows 12-17, 19-24, 26-31).
- Row 10 of Task sheet (years in columns H-L).
- Data sheet rows 21-38 layout: which column holds the series code, which row holds years, and where the numeric data lives.
- The three blocks on Task: rows 12-17 (block 1), 19-24 (block 2), 26-31 (block 3) — what metric does each represent?
- Rows 35-40: Net container flow formula.
- Rows 42-47: statistics (min, max, median, mean, 25th pctl, 75th pctl).
- Row 50: weighted mean.

## 1 – Write lookup formulas in H12:L17, H19:L24, H26:L31

Use openpyxl to write Excel formulas (strings starting with `=`) into each yellow cell.

Pattern to use (INDEX/MATCH is safest):
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```
Adjust the ranges based on what you discovered:
- `<data_range>`: the rectangular block on Data sheet rows 21-38 that holds numeric values.
- `<series_code_column>`: the column on Data that holds series codes (same column as column D references).
- `<year_header_row>`: the row on Data that holds year headers.

Make sure:
- The row reference for the series code uses `$D12` (column-absolute) so it stays in column D as you go across columns H-L.
- The column reference for the year uses `H$10` (row-absolute) so it stays in row 10 as you go down rows.
- Apply the same pattern for all three blocks, adjusting row numbers (12-17, 19-24, 26-31).

## 2 – Net container flow in H35:L40

Identify which block is "Loaded Containers Inbound" (likely rows 12-17), which is "Loaded Containers Outbound" (likely rows 19-24), and which is "Terminal Throughput Capacity" (rows 26-31). Confirm from the Task sheet labels.

For each cell in H35:L40:
```
=(H12 - H19) / H26 * 100
```
Adjust row references so each port row maps correctly (row 35↔rows 12,19,26; row 36↔rows 13,20,27; etc.).

## 3 – Statistics in H42:L47

For each column (H through L):
- H42: `=MIN(H35:H40)`
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)`
- H46: `=PERCENTILE(H35:H40,0.25)`  ← Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC` (cross-task failures show #NAME? from unrecognized function names; check which function the verifier/Excel engine supports — `PERCENTILE` is the safest legacy name; if the test expects `.INC` variant, use that)
- H47: `=PERCENTILE(H35:H40,0.75)`

**Important**: Verify the order (min/max/median/mean/25th/75th) matches the labels in column A-G of rows 42-47 on the Task sheet. Adjust if needed.

## 4 – Weighted mean in H50:L50

```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the net-container-flow percentages using Terminal Throughput Capacity as weights.

## 5 – Save

```python
import shutil, os
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## 6 – Validate

Re-open the saved file and print all formulas in the modified cells to confirm they are present and well-formed:
```python
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb2['Task']
for r in range(12, 18):
    for c in range(8, 13):  # H=8, L=12
        print(f"{ws.cell(r,c).coordinate}: {ws.cell(r,c).value}")
# repeat for rows 19-24, 26-31, 35-40, 42-47, 50
```

Then run the test suite:
```bash
cd /root && python -m pytest tests/ -v 2>&1 | tail -80
```

If any cells show `#NAME?`, check whether the function name needs adjustment (e.g., `PERCENTILE.INC` vs `PERCENTILE`). If cells return `None`, the formula was not written. Fix and re-save.

If the verifier evaluates formulas (data_only=True), it needs a calc engine. Check the test code — if it uses openpyxl with data_only=True, formulas won't resolve. In that case you may need to also write computed numeric values. But first check the test to see how it reads cells.

Keep existing formatting. Do not add sheets, macros, VBA, external links, or helper tabs.

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