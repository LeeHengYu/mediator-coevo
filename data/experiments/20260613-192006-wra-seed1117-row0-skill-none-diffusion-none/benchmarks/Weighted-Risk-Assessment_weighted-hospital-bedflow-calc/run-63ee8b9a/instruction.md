# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## Phase 0 – Inspect the workbook
```python
import openpyxl, os, json
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
print('Sheet names:', wb.sheetnames)

task = wb['Task']
data = wb['Data']

# Print row 10 headers (years) in columns H-L
print('Row 10 (years):', [task.cell(row=10, column=c).value for c in range(8,13)])

# Print column D series codes for rows 12-17, 19-24, 26-31
for label, rng in [('Block1', range(12,18)), ('Block2', range(19,25)), ('Block3', range(26,32))]:
    print(f'{label} series codes (col D):', [(r, task.cell(row=r, column=4).value) for r in rng])

# Print row 35-40 labels
print('Rows 35-40 labels:', [(r, task.cell(row=r, column=4).value) for r in range(35,41)])
# Print row 42-47 labels
print('Rows 42-47 labels:', [(r, task.cell(row=r, column=4).value) for r in range(42,48)])
# Print row 50 label
print('Row 50 label:', task.cell(row=50, column=4).value)

# Inspect Data sheet structure – rows 21-38
print('\nData sheet row 21-38, first 15 cols:')
for r in range(21,39):
    print(r, [data.cell(row=r, column=c).value for c in range(1,16)])

# Also check Data sheet header row (likely row 20 or row 1)
for r in [1,2,19,20]:
    print(f'Data row {r}:', [data.cell(row=r, column=c).value for c in range(1,16)])

wb.close()
```
Run this and study the output carefully before proceeding.

## Phase 1 – Write the formulas

Using openpyxl, open the workbook again (without data_only so formulas are preserved), and write formulas as described below. All formulas must start with '='.

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, use an INDEX/MATCH/MATCH formula that:
- Looks up the series code from column D of the **current row** in the Data sheet rows 21-38 (the first column of Data that contains the series codes).
- Looks up the year from row 10 of the Task sheet in the Data sheet header row (the row that contains years).
- Returns the intersection value.

Determine the exact column letters and row numbers from the Phase 0 inspection. The pattern is:
```
=INDEX(Data!$B$21:$P$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$P$20, 0))
```
Adjust the ranges based on actual layout discovered in Phase 0. The key anchor: column D has the series code, row 10 has the year, Data rows 21-38 have the source records.

Use absolute references for the Data ranges and mixed references ($D for column, $ for row 10) so the formula can be placed in each cell with appropriate adjustments. Since openpyxl doesn't auto-adjust references when assigning to cells, you must construct each formula with the correct row reference for column D and the correct column reference for row 10.

Concretely, loop over each block and each cell:
```python
for row in range(12, 18):  # and 19-24, 26-31
    for col_idx, col_letter in [(8,'H'),(9,'I'),(10,'J'),(11,'K'),(12,'L')]:
        formula = f'=INDEX(Data!$B$21:$P$38,MATCH($D{row},Data!$A$21:$A$38,0),MATCH({col_letter}$10,Data!$B$20:$P$20,0))'
        task.cell(row=row, column=col_idx).value = formula
```
Again, adjust range references based on Phase 0 findings.

### Step 2 – Net patient flow in H35:L40
For each hospital row (35-40), the formula is:
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to Patient Admissions, H19 to Patient Discharges, H26 to Effective Bed Capacity, for the same hospital in the same column. Adjust row offsets: row 35 maps to rows 12, 19, 26; row 36 maps to 13, 20, 27; etc.

```python
for i in range(6):
    for col_idx, col_letter in [(8,'H'),(9,'I'),(10,'J'),(11,'K'),(12,'L')]:
        adm_row = 12 + i
        dis_row = 19 + i
        cap_row = 26 + i
        net_row = 35 + i
        formula = f'=({col_letter}{adm_row}-{col_letter}{dis_row})/{col_letter}{cap_row}*100'
        task.cell(row=net_row, column=col_idx).value = formula
```

### Step 2 continued – Statistics in H42:L47
Rows 42-47 are: MIN, MAX, MEDIAN, AVERAGE, 25th percentile, 75th percentile (verify labels from Phase 0).

```python
stat_formulas = [
    'MIN',      # row 42
    'MAX',      # row 43
    'MEDIAN',   # row 44
    'AVERAGE',  # row 45
    'PERCENTILE', # row 46 – 25th
    'PERCENTILE', # row 47 – 75th
]
for col_idx, col_letter in [(8,'H'),(9,'I'),(10,'J'),(11,'K'),(12,'L')]:
    rng = f'{col_letter}35:{col_letter}40'
    task.cell(row=42, column=col_idx).value = f'=MIN({rng})'
    task.cell(row=43, column=col_idx).value = f'=MAX({rng})'
    task.cell(row=44, column=col_idx).value = f'=MEDIAN({rng})'
    task.cell(row=45, column=col_idx).value = f'=AVERAGE({rng})'
    task.cell(row=46, column=col_idx).value = f'=PERCENTILE({rng},0.25)'
    task.cell(row=47, column=col_idx).value = f'=PERCENTILE({rng},0.75)'
```

**IMPORTANT**: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) based on cross-task feedback showing #NAME? errors with dotted variants.

### Step 3 – Weighted mean in H50:L50
```python
for col_idx, col_letter in [(8,'H'),(9,'I'),(10,'J'),(11,'K'),(12,'L')]:
    vals = f'{col_letter}35:{col_letter}40'
    wts  = f'{col_letter}26:{col_letter}31'
    formula = f'=SUMPRODUCT({vals},{wts})/SUM({wts})'
    task.cell(row=50, column=col_idx).value = formula
```

## Phase 2 – Save
```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
wb.close()
```

## Phase 3 – Verify
Reopen the saved file and confirm:
1. The 'Task' sheet exists and cells H12, H19, H26, H35, H42, H50 all contain strings starting with '='.
2. Print a sample of 5-6 formula cells to confirm correctness.
3. Confirm no extra sheets were added.

```python
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
t = wb2['Task']
for (r,c) in [(12,8),(19,8),(26,8),(35,8),(42,8),(46,8),(47,8),(50,8)]:
    print(f'Row {r}, Col {c}: {t.cell(row=r, column=c).value}')
print('Sheets:', wb2.sheetnames)
wb2.close()
```

**Critical reminders:**
- Adjust ALL range references based on what you discover in Phase 0. Do NOT blindly use the example ranges.
- Every formula cell value must be a string starting with '='.
- Do NOT add any sheets, macros, or external links.
- The stat labels in rows 42-47 may be in a different order than assumed – check Phase 0 output and adjust accordingly.
- Use `PERCENTILE` not `PERCENTILE.INC`.

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