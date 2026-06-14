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

# Print Task sheet structure: rows 1-55, columns A-M
for row in ws_task.iter_rows(min_row=1, max_row=55, min_col=1, max_col=13, values_only=False):
    print([(c.coordinate, c.value) for c in row])

# Print Data sheet structure: rows 1-45, columns A-Z (or however wide)
for row in ws_data.iter_rows(min_row=1, max_row=45, min_col=1, max_col=26, values_only=False):
    print([(c.coordinate, c.value) for c in row])
```

Carefully note:
- What is in column D for rows 12-17, 19-24, 26-31 (the series codes)
- What is in row 10 columns H-L (the years)
- What is in Data sheet rows 21-38 (the source data layout — which row/column has what)
- What the three blocks represent (likely Latency Budget Preserved, Latency Budget Consumed, Covered Request Capacity)
- The exact labels in rows 35-40, 42-47, 50

## 1. Step 1 — Populate H12:L31 with lookup formulas

Using openpyxl, write formulas into the yellow cells. For each cell in ranges H12:L17, H19:L24, H26:L31:

The formula pattern should use INDEX/MATCH (most reliable in openpyxl):
```
=INDEX(Data!$A$21:$ZZ$38, MATCH($D{row}, Data!$A$21:$A$38, 0), MATCH({year_cell}, Data!$A$21:$ZZ$21, 0))
```

Where:
- `$D{row}` is the series code in column D of the current row on the Task sheet (use absolute row reference for the lookup value but relative to the current row)
- `{year_cell}` is the year header cell in row 10 (e.g., H$10, I$10, etc.)
- The Data range for rows should cover rows 21:38
- The Data range for the header row should be row 21 (or whichever row contains the column headers — verify from inspection)

IMPORTANT: Before writing formulas, verify from your inspection:
- Which row in Data contains the header/year values (it might be row 21 or a different row)
- Which column in Data contains the series codes
- Adjust the INDEX/MATCH ranges accordingly

The exact formula for cell H12 should look something like:
```
=INDEX(Data!$B$22:$ZZ$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$ZZ$21, 0))
```
But adjust based on actual Data sheet layout.

Alternatively, if the layout is suitable, use:
```
=INDEX(Data!$A$21:$ZZ$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$A$21:$ZZ$21, 0))
```

Write these formulas for all 60 cells (3 blocks × 6 rows × 5 columns).

## 2. Step 2a — Net SLA Buffer in H35:L40

For each cell in H35:L40, the formula computes:
`(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`

Identify which block is which:
- If H12:L17 = Latency Budget Preserved, H19:L24 = Latency Budget Consumed, H26:L31 = Covered Request Capacity
- Then for H35: `=(H12-H19)/H26*100`
- For I35: `=(I12-I19)/I26*100`
- etc., matching the row offsets (row 35↔row 12/19/26, row 36↔row 13/20/27, ...)

Verify the block-to-metric mapping from the labels you inspected. Write the formulas accordingly.

## 2b. Step 2b — Summary statistics in H42:L47

For each column (H through L), compute over the 6 Net SLA Buffer values (rows 35-40):
- Row 42 (minimum): `=MIN(H35:H40)`
- Row 43 (maximum): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)`

Verify the row-to-statistic mapping from the labels. The order (min, max, median, mean, 25th, 75th) must match the labels in column A/B/C/D.

## 3. Step 3 — Weighted mean in H50:L50

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

This computes the weighted mean of Net SLA Buffer percentages weighted by Covered Request Capacity.

## 4. Save

```python
wb.save('/root/output/result.xlsx')
```

## 5. Verification

Reload the saved file and verify:
```python
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb2['Task']
# Check that formulas exist in all required cells
for r in range(12, 18):
    for c in ['H','I','J','K','L']:
        cell = ws[f'{c}{r}']
        print(f'{c}{r}: {cell.value}')
# Repeat for rows 19-24, 26-31, 35-40, 42-47, 50
for r in list(range(19,25)) + list(range(26,32)) + list(range(35,41)) + list(range(42,48)) + [50]:
    for c in ['H','I','J','K','L']:
        cell = ws[f'{c}{r}']
        print(f'{c}{r}: {cell.value}')
```

Confirm:
- All 60 lookup cells contain INDEX+MATCH formulas referencing Data sheet
- All 30 Net SLA Buffer cells contain the correct arithmetic formula
- All 30 summary statistic cells contain the correct function
- All 5 weighted mean cells contain SUMPRODUCT formulas
- No new sheets were added
- File is saved at /root/output/result.xlsx

IMPORTANT NOTES:
- You MUST inspect the workbook first before writing any formulas. The exact row/column references depend on the actual layout.
- Use openpyxl to write formulas as strings (e.g., `ws['H12'] = '=INDEX(...)'`)
- Do NOT use data_only mode when loading for writing
- Preserve all existing formatting — do not clear or overwrite cells outside the specified ranges
- When writing formulas with openpyxl, make sure the formula string starts with '='

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