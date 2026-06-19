# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0 – Inspect the workbook
```python
import openpyxl, os, shutil
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
print('Sheet names:', wb.sheetnames)

# Inspect Data sheet layout
ds = wb['Data']
print('Data header row 21:', [ds.cell(row=21, column=c).value for c in range(1, 20)])
print('Data rows 22-38 col A:', [ds.cell(row=r, column=1).value for r in range(22, 39)])

# Inspect Task sheet layout
ts = wb['Task']
print('Task row 10 (years):', [ts.cell(row=10, column=c).value for c in range(1, 20)])
print('Task col D (series codes):', [ts.cell(row=r, column=4).value for r in range(12, 32)])
print('Task col D rows 35-40:', [ts.cell(row=r, column=4).value for r in range(35, 41)])
print('Task row 42-47 col G labels:', [ts.cell(row=r, column=7).value for r in range(42, 48)])
print('Task row 50 label:', ts.cell(row=50, column=7).value)
print('Task H12 current:', ts.cell(row=12, column=8).value)
wb.close()
```
Record the exact column letters/numbers for the Data sheet's series-code column and years header row. Confirm which columns in Task correspond to H–L (columns 8–12) and which row holds years (row 10).

## 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31

Using openpyxl, write INDEX/MATCH formulas into each cell. The pattern for cell at (row, col) should be:

```
=INDEX(Data!$B$22:$M$38, MATCH($D{row}, Data!$A$22:$A$38, 0), MATCH({col_letter}$10, Data!$B$21:$M$21, 0))
```

Adjust the absolute references ($B$22:$M$38, $A$22:$A$38, $B$21:$M$21) based on what you discover in step 0 about the actual data layout. The key anchors:
- `$D{row}` – the series code in column D of the current Task row (row-absolute within the formula, but the row number changes per cell)
- `{col_letter}$10` – the year in row 10 of the current column (H10, I10, … L10)

Iterate over the three blocks:
- Rows 12–17, columns H–L (8–12)
- Rows 19–24, columns H–L
- Rows 26–31, columns H–L

Use `openpyxl.utils.get_column_letter` for the column letter.

## 2 – Net capacity headroom (H35:L40)

For each of the six hospital clusters (rows 35–40) and each year column (H–L), write a formula:
```
=({H12_ref} - {H19_ref}) / {H26_ref} * 100
```
where:
- H12_ref corresponds to Available Care Slots (rows 12–17)
- H19_ref corresponds to Occupied Care Slots (rows 19–24)
- H26_ref corresponds to Staffed Bed Capacity (rows 26–31)

The row offset: row 35 maps to rows 12, 19, 26; row 36 maps to 13, 20, 27; etc.

So for Task cell at (r, c) where r in [35..40], c in [8..12]:
```
=({col}{r-23} - {col}{r-16}) / {col}{r-9} * 100
```
Verify: row 35 → (col_letter + "12" - col_letter + "19") / col_letter + "26" * 100 → offsets: 35-23=12, 35-16=19, 35-9=26. ✓

## 3 – Summary statistics (H42:L47)

For each column (H–L), write these formulas referencing the headroom block {col}35:{col}40:
- Row 42: `=MIN({col}35:{col}40)`
- Row 43: `=MAX({col}35:{col}40)`
- Row 44: `=MEDIAN({col}35:{col}40)`
- Row 45: `=AVERAGE({col}35:{col}40)`
- Row 46: `=PERCENTILE({col}35:{col}40, 0.25)`
- Row 47: `=PERCENTILE({col}35:{col}40, 0.75)`

Confirm the order by checking the labels in column G rows 42–47 from step 0. Adjust the mapping if the labels differ (e.g., if MIN is not row 42).

## 4 – Weighted mean (H50:L50)

For each column (H–L):
```
=SUMPRODUCT({col}35:{col}40, {col}26:{col}31) / SUM({col}26:{col}31)
```

## 5 – Save

```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
wb.close()
```

## 6 – Validate

Reopen the saved file and print the formula strings in a few representative cells to confirm they are correct:
- H12, L17 (lookup block boundaries)
- H35, L40 (headroom block boundaries)
- H42, H47 (stats)
- H50, L50 (weighted mean)

Also run any test script if present:
```bash
ls /root/*.py /root/test*.py 2>/dev/null
# If test_output.py exists:
cd /root && python -m pytest test_output.py -v
```

## Critical notes
- Do NOT use `data_only=True` when loading; formulas must be preserved.
- Do NOT add new sheets, macros, or VBA.
- Do NOT alter existing formatting.
- The previous failed run on a sibling task (weighted-hospital-bedflow-calc) failed because cells were left empty (None). Ensure every cell in the target ranges gets a formula string.
- The previous successful run on this exact task used INDEX/MATCH with $D12 and H$10 style references. Follow the same pattern.

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