# Task Instruction

Execute the following steps in a single Python script to produce `/root/output/result.xlsx`.

## Preliminary
```
import os, shutil
from openpyxl import load_workbook
os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')
wb = load_workbook('/root/output/result.xlsx')
ws = wb['Task']
data_ws = wb['Data']
```

## Step 0 – Inspect the workbook
1. Print rows 10-11 of the Task sheet (columns A-L) to see the header/year row.
2. Print rows 12-31 of the Task sheet (columns A-L) to see the series codes and existing content.
3. Print rows 35-50 of the Task sheet (columns A-L) to see the Net patient flow area, stats area, and weighted mean row.
4. Print rows 21-38 of the Data sheet (columns A-Z or however wide it goes) to understand the data layout (orientation, whether series codes are in a row or column, where years appear).
5. Print row 1 (or the header rows) of the Data sheet to understand column headers.
6. Also print the range H26:L31 on Task to confirm the Effective Bed Capacity block location.

Do NOT proceed to formula writing until you have printed and understood all of the above. Post the printed output so we can reason about it.

## Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write INDEX/MATCH formulas into the yellow cells. The pattern for each cell should be:
```
=INDEX(Data!<value_range>, MATCH(<series_code_cell>, Data!<series_code_column>, 0), MATCH(<year_cell>, Data!<year_row>, 0))
```
where:
- `<series_code_cell>` is the absolute reference to column D of the current row on the Task sheet (e.g., `$D12`).
- `<year_cell>` is the absolute reference to the year in row 10 for the current column (e.g., `H$10`).
- `<value_range>`, `<series_code_column>`, and `<year_row>` come from the Data sheet layout you inspected.

IMPORTANT: The Data sheet rows 21:38 contain the source data. Determine whether:
- Series codes are in a column (e.g., column A or B) and years are in a row (e.g., row 20 or row 21).
- Or the layout is transposed.

Adjust the INDEX/MATCH accordingly. Use absolute references where needed so formulas can be filled across the range.

Assign each formula as a string to `ws.cell(row=r, col=c).value`. For example:
```python
for r in range(12, 18):  # rows 12-17
    for c in range(8, 13):  # columns H(8) through L(12)
        series_ref = f'$D{r}'
        year_ref = f'{get_column_letter(c)}$10'
        formula = f'=INDEX(Data!$C$21:$G$38,MATCH({series_ref},Data!$A$21:$A$38,0),MATCH({year_ref},Data!$C$20:$G$20,0))'
        ws.cell(row=r, column=c).value = formula
```
(Adjust the exact Data sheet references based on your inspection.)

Repeat for rows 19-24 and 26-31 with the same pattern.

## Step 2 – Net patient flow in H35:L40

The formula for each cell is:
```
=(<Admissions_cell> - <Discharges_cell>) / <Capacity_cell> * 100
```
where Admissions are in H12:L17, Discharges in H19:L24, Capacity in H26:L31. The hospital order in rows 35-40 should match the order in rows 12-17. Verify by inspecting column D or column B of rows 35-40 vs 12-17.

For example:
```python
for i in range(6):
    for c in range(8, 13):
        col_letter = get_column_letter(c)
        adm = f'{col_letter}{12+i}'
        dis = f'{col_letter}{19+i}'
        cap = f'{col_letter}{26+i}'
        ws.cell(row=35+i, column=c).value = f'=({adm}-{dis})/{cap}*100'
```

## Step 3 – Statistics in H42:L47

For each column (H through L), compute these six statistics over the 6 Net patient flow values (e.g., H35:H40):
- Row 42: MIN
- Row 43: MAX  
- Row 44: MEDIAN
- Row 45: AVERAGE
- Row 46: PERCENTILE (25th) – use `PERCENTILE` not `PERCENTILE.INC` to avoid #NAME? errors in some engines
- Row 47: PERCENTILE (75th)

BUT FIRST: Inspect what labels are in column B or C of rows 42-47 to determine the correct order. The order above is a guess; use whatever order the sheet specifies.

For PERCENTILE, use `PERCENTILE(<range>,0.25)` and `PERCENTILE(<range>,0.75)`. If the sheet labels say "25th percentile" / "75th percentile", confirm which rows they are in.

Note from cross-task context: `#NAME?` errors occurred in a similar task when using `PERCENTILE.INC`. Prefer `PERCENTILE` instead.

```python
for c in range(8, 13):
    col_letter = get_column_letter(c)
    rng = f'{col_letter}35:{col_letter}40'
    ws.cell(row=42, column=c).value = f'=MIN({rng})'
    ws.cell(row=43, column=c).value = f'=MAX({rng})'
    ws.cell(row=44, column=c).value = f'=MEDIAN({rng})'
    ws.cell(row=45, column=c).value = f'=AVERAGE({rng})'
    ws.cell(row=46, column=c).value = f'=PERCENTILE({rng},0.25)'
    ws.cell(row=47, column=c).value = f'=PERCENTILE({rng},0.75)'
```
(Adjust row numbers based on inspection.)

## Step 4 – Weighted mean in H50:L50

For each column:
```
=SUMPRODUCT(<net_flow_range>, <capacity_range>) / SUM(<capacity_range>)
```
where net_flow_range is the Step 2 percentages (e.g., H35:H40) and capacity_range is the Effective Bed Capacity block (e.g., H26:H31).

```python
for c in range(8, 13):
    col_letter = get_column_letter(c)
    flow = f'{col_letter}35:{col_letter}40'
    cap = f'{col_letter}26:{col_letter}31'
    ws.cell(row=50, column=c).value = f'=SUMPRODUCT({flow},{cap})/SUM({cap})'
```

## Step 5 – Save
```python
wb.save('/root/output/result.xlsx')
```

## Step 6 – Verify
Reload the workbook and print the values of key cells to confirm formulas were written:
```python
wb2 = load_workbook('/root/output/result.xlsx')
ws2 = wb2['Task']
for r in [12, 17, 19, 24, 26, 31, 35, 40, 42, 47, 50]:
    for c in range(8, 13):
        print(f'{get_column_letter(c)}{r}: {ws2.cell(row=r, column=c).value}')
```
Confirm that each cell contains a string starting with '='. If any cell is None, debug and fix before finishing.

## Critical reminders
- Do the inspection FIRST. Print the Data sheet layout before writing any formulas.
- Assign formulas as strings to `.value` (e.g., `cell.value = '=INDEX(...)'`).
- Do NOT use `data_only=True` when loading.
- Do NOT add sheets, macros, or VBA.
- Adjust all row/column references based on actual inspection results, not the guesses above.

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