# Task Instruction

Execute the following steps exactly.

## Step 0 – Inspect the workbook
```bash
cd /root && python3 << 'PYEOF'
import openpyxl, json
wb = openpyxl.load_workbook('data/workbook.xlsx')
for name in wb.sheetnames:
    print(f'Sheet: {name}')
ts = wb['Task']
# Print row 10 (header row with years)
print('Row 10:', [(c.coordinate, c.value) for c in ts[10]])
# Print column D rows 12-31 (series codes)
for r in range(12, 32):
    print(f'D{r}:', ts[f'D{r}'].value, '  G{r}:', ts[f'G{r}'].value)
# Print row 35-40 labels
for r in range(35, 41):
    print(f'Row {r} labels:', ts[f'B{r}'].value, ts[f'C{r}'].value, ts[f'D{r}'].value)
# Print row 42-47 labels
for r in range(42, 48):
    print(f'Row {r} labels:', ts[f'B{r}'].value, ts[f'C{r}'].value, ts[f'D{r}'].value, ts[f'G{r}'].value)
# Print row 50 labels
print('Row 50:', ts['B50'].value, ts['C50'].value, ts['D50'].value, ts['G50'].value)
# Check Data sheet structure
ds = wb['Data']
print('\nData sheet rows 19-40 col A-F:')
for r in range(19, 41):
    vals = [ds.cell(r, c).value for c in range(1, 7)]
    print(f'  Row {r}: {vals}')
print('Data sheet row 20 (header?):', [ds.cell(20, c).value for c in range(1, 20)])
print('Data sheet row 21:', [ds.cell(21, c).value for c in range(1, 20)])
print('Data sheet row 38:', [ds.cell(38, c).value for c in range(1, 20)])
wb.close()
PYEOF
```

Read the output carefully. Identify:
- The exact column letters in Data rows 21-38 that hold the series code (lookup key) and the year columns (numeric years).
- The exact column letters in Task row 10 that hold years for H-L (columns 8-12).
- The exact series codes in Task column D for rows 12-17, 19-24, 26-31.
- The labels in rows 42-47 (min, max, median, mean, 25th, 75th percentile).

## Step 1 – Write the workbook with formulas

After inspecting, run a Python script that:
1. Opens `/root/data/workbook.xlsx` with `openpyxl.load_workbook('data/workbook.xlsx')`.
2. For each cell in H12:L17, H19:L24, H26:L31 on sheet `Task`, writes an INDEX/MATCH formula. The formula pattern should be:
   `=INDEX(Data!<year_col_in_data>$21:Data!<year_col_in_data>$38, MATCH($D<row>, Data!$<code_col>$21:Data!$<code_col>$38, 0))`
   BUT – you must first determine from the inspection which column in Data holds the series code and which columns hold year data. If Data has years in a header row, you may instead use a two-dimensional INDEX/MATCH:
   `=INDEX(Data!<data_range>, MATCH($D<row>, Data!<code_column_range>, 0), MATCH(<Task_year_cell>, Data!<year_header_range>, 0))`
   Choose whichever INDEX+MATCH pattern fits the Data layout discovered in Step 0.

3. For H35:L40 (Net patient flow for 6 hospitals), write:
   `=(<admissions_cell> - <discharges_cell>) / <bed_capacity_cell> * 100`
   where admissions are in H12:L17, discharges in H19:L24, bed capacity in H26:L31. Map row 35→row 12/19/26, row 36→row 13/20/27, etc.

4. For rows 42-47 (statistics on H35:L40), determine the exact stat for each row from the labels found in Step 0. Write column-wise formulas:
   - MIN: `=MIN(H35:H40)` (or the appropriate column)
   - MAX: `=MAX(H35:H40)`
   - MEDIAN: `=MEDIAN(H35:H40)`
   - AVERAGE: `=AVERAGE(H35:H40)`
   - 25th percentile: `=PERCENTILE(H35:H40,0.25)`   ← use PERCENTILE not PERCENTILE.INC
   - 75th percentile: `=PERCENTILE(H35:H40,0.75)`

5. For H50:L50 (weighted mean), write:
   `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)` (adjust column letter for each column H-L).

6. Save to `/root/output/result.xlsx`. Create `/root/output/` directory first if needed.

IMPORTANT NOTES:
- Use `wb.save(...)` not `wb.close()` before closing.
- All formulas must be strings starting with `=`.
- Do NOT use `data_only=True` when loading.
- Do NOT add any new sheets.
- Preserve all existing formatting.
- After saving, re-open the file and print the values/formulas of a sample of cells (e.g., H12, H35, H42, H50) to confirm they contain formula strings, not None.

## Step 2 – Validate
```bash
cd /root && python3 << 'PYEOF'
import openpyxl
wb = openpyxl.load_workbook('output/result.xlsx')
ts = wb['Task']
for coord in ['H12','L17','H19','L24','H26','L31','H35','L40','H42','L47','H50','L50']:
    print(f'{coord}: {ts[coord].value}')
wb.close()
PYEOF
```
Every printed cell must show a formula string (starting with `=`), NOT None. If any cell is None, diagnose and fix before finishing.

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