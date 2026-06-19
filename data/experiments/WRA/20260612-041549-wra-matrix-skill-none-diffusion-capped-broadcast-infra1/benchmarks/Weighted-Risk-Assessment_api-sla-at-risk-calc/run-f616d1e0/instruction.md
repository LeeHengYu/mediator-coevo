# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0 – Inspect the workbook
```bash
pip install openpyxl
```
```python
import openpyxl, pprint
wb = openpyxl.load_workbook('/root/data/workbook.xlsx', data_only=True)
for s in wb.sheetnames:
    print(f'--- {s} ---')
    ws = wb[s]
    print(f'  rows={ws.min_row}-{ws.max_row}  cols={ws.min_column}-{ws.max_column}')
# On sheet Task, print rows 10-50 cols D-L to understand layout
ws = wb['Task']
for r in range(10, 51):
    vals = []
    for c in range(4, 13):  # D=4 .. L=12
        cell = ws.cell(r, c)
        vals.append(f'{cell.coordinate}={cell.value}')
    print(' | '.join(vals))
# On sheet Data, print rows 19-40 to understand lookup source
ws2 = wb['Data']
for r in range(19, 41):
    vals = []
    for c in range(1, ws2.max_column+1):
        cell = ws2.cell(r, c)
        vals.append(f'{cell.coordinate}={cell.value}')
    print(' | '.join(vals))
wb.close()
```
Carefully study the output to determine:
- The series codes in column D for rows 12-17, 19-24, 26-31.
- The years in row 10 for columns H-L.
- The layout of Data!rows 21:38 (header row, key column, data columns).
- What the three blocks represent (Latency Budget Preserved, Latency Budget Consumed, Covered Request Capacity – or whatever the actual labels are).
- The exact row/column references needed for MATCH and the lookup functions.

## 1 – Write the formulas
Open the workbook with openpyxl (without data_only) and write formulas into the cells.

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each yellow cell at row `r`, column `c` (H=8 … L=12):
- The series code is in `D{r}` on sheet Task.
- The year is in `{col_letter}10` on sheet Task (same column as the cell).
- The lookup source is Data!rows 21:38.

Use INDEX/MATCH (safest in openpyxl):
```
=INDEX(Data!<data_area>, MATCH(D{r}, Data!<key_column>, 0), MATCH({col}10, Data!<header_row>, 0))
```
Adjust the exact ranges based on what you discovered in step 0. Make sure:
- `<data_area>` covers the numeric block (not headers/keys).
- `<key_column>` is the column of series codes in Data.
- `<header_row>` is the row of year headers in Data.
- Use absolute references where appropriate to avoid drift.

Write every formula as a string starting with '=' into the cell.

### Step 2 – Net SLA buffer (H35:L40) and statistics (H42:L47)
For each service row `i` (0..5), the three input blocks are at rows 12+i, 19+i, 26+i respectively (for the same column). Write:
```
=(H{12+i} - H{19+i}) / H{26+i} * 100
```
(adjusting column letter for each column H-L).

For statistics in H42:L47, for each column `col` (H-L):
- Row 42: `=MIN({col}35:{col}40)`
- Row 43: `=MAX({col}35:{col}40)`
- Row 44: `=MEDIAN({col}35:{col}40)`
- Row 45: `=AVERAGE({col}35:{col}40)`
- Row 46: `=_xlfn.PERCENTILE.INC({col}35:{col}40,0.25)`
- Row 47: `=_xlfn.PERCENTILE.INC({col}35:{col}40,0.75)`

IMPORTANT: Use `_xlfn.PERCENTILE.INC` prefix for the percentile functions – this is required by openpyxl for Excel's newer function names.

Check the actual row labels (min/max/median/mean/25th/75th) in the workbook to confirm which statistic goes in which row. Adjust the row assignments accordingly.

### Step 3 – Weighted mean (H50:L50)
For each column `col` (H-L):
```
=SUMPRODUCT({col}35:{col}40, {col}26:{col}31) / SUM({col}26:{col}31)
```

## 2 – Save
```python
import os
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## 3 – Validate
Re-open the saved file with data_only=False and spot-check:
- Cells H12, L17, H19, L24, H26, L31 contain formula strings (start with '=').
- Cells H35, L40 contain formula strings.
- Cells H42, H47 contain formula strings with appropriate function names.
- Cell H50 contains a SUMPRODUCT formula.
- No cells in the target ranges are None or empty.
- Sheet count is unchanged; no new sheets exist.

Also re-open with data_only=True to check that openpyxl cached values are present (they may be None since openpyxl doesn't evaluate, which is fine – the formulas are what matter).

If any target cell is None or missing a formula, debug and fix before finishing.

## Key cautions
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Use `_xlfn.PERCENTILE.INC` for percentile functions.
- Verify the exact row labels and block boundaries from the inspection before writing formulas.
- If the statistics rows are ordered differently than min/max/median/mean/25th/75th, match the actual labels in the workbook.

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