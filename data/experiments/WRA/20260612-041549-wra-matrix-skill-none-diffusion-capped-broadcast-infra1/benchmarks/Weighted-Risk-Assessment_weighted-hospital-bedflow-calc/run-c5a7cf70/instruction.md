# Task Instruction

Execute the following steps precisely to produce /root/output/result.xlsx.

## Step 0 – Inspect the workbook

```python
import openpyxl, os, json

wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
print('Sheet names:', wb.sheetnames)

ts = wb['Task']
ds = wb['Data']

# Print Task sheet layout rows 1-55, cols A-M
print('\n=== Task sheet ===')
for row in ts.iter_rows(min_row=1, max_row=55, min_col=1, max_col=13, values_only=False):
    vals = [(c.coordinate, c.value) for c in row]
    print(vals)

# Print Data sheet rows 1-45, cols A-Z (enough to find structure)
print('\n=== Data sheet ===')
for row in ds.iter_rows(min_row=1, max_row=45, min_col=1, max_col=26, values_only=False):
    vals = [(c.coordinate, c.value) for c in row if c.value is not None]
    if vals:
        print(vals)
```

Read every printed line carefully. Identify:
- Column D series codes for rows 12-17, 19-24, 26-31 on Task sheet.
- Row 10 year headers in columns H-L on Task sheet.
- Data sheet row 21-38 layout: which row holds which series code, which column holds which year, and which row/column holds the header labels.
- The exact cell references for the Data lookup range (anchor row, anchor column, last row, last column).
- The labels in rows 35-40 (hospitals), rows 42-47 (statistics), row 50.
- Any existing formulas or values already in the workbook.

## Step 1 – Write lookup formulas in H12:L31

Use `INDEX(MATCH,MATCH)` pattern. Based on the Data sheet inspection:
- Determine the data array range (e.g., Data!$B$22:$Z$38 or similar – use actual coordinates from inspection).
- Determine the row header range (series codes column in Data sheet, e.g., Data!$A$22:$A$38).
- Determine the column header range (year row in Data sheet, e.g., Data!$B$21:$Z$21).

For each cell in H12:L17, H19:L24, H26:L31 on Task sheet, write a formula string like:
`=INDEX(Data!$B$22:$Z$38,MATCH($D12,Data!$A$22:$A$38,0),MATCH(H$10,Data!$B$21:$Z$21,0))`

Adjust the exact ranges based on what you found in Step 0. The $D reference must be the series-code column on Task sheet; H$10 must be the year-header row. Use absolute references for the Data ranges and mixed references ($D for column lock, $10 for row lock) so formulas copy correctly across the block.

Write formulas using openpyxl by assigning formula strings (starting with '=') to each cell. Do NOT use data_only mode. Example:
```python
ts['H12'] = '=INDEX(Data!$B$22:$Z$38,MATCH($D12,Data!$A$22:$A$38,0),MATCH(H$10,Data!$B$21:$Z$21,0))'
```

Loop over all 60 cells (6 rows × 5 cols × 3 blocks).

## Step 2 – Net patient flow formulas in H35:L40

For each hospital row i (35-40), the formula references the corresponding rows in the three blocks:
- Patient Admissions: rows 12-17 (row 12 corresponds to row 35, etc.)
- Patient Discharges: rows 19-24 (row 19 corresponds to row 35, etc.)
- Effective Bed Capacity: rows 26-31 (row 26 corresponds to row 35, etc.)

Formula for cell H35:
`=(H12-H19)/H26*100`

Generalize for each cell in H35:L40 using the correct row offsets.

## Step 3 – Statistics formulas in H42:L47

For each column (H through L):
- Row 42 (Min): `=MIN(H35:H40)`
- Row 43 (Max): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`  (NOT PERCENTILE.INC)
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`  (NOT PERCENTILE.INC)

IMPORTANT: Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC` – the dotted versions cause #NAME? errors in some engines. Similarly use `MEDIAN`, `MIN`, `MAX`, `AVERAGE` (all standard).

Verify the row assignments (42=min, 43=max, etc.) against the labels you found in Step 0. Adjust if the labels are in a different order.

## Step 4 – Weighted mean in H50:L50

For each column (H through L):
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This computes the weighted mean of Net patient flow using Effective Bed Capacity as weights.

## Step 5 – Save and verify

```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

Then reload and verify:
```python
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
ts2 = wb2['Task']

# Check formulas are strings starting with '='
for coord in ['H12','L17','H19','L24','H26','L31','H35','L40','H42','H47','H50','L50']:
    v = ts2[coord].value
    print(f'{coord}: {repr(v)}')
    assert isinstance(v, str) and v.startswith('='), f'{coord} is not a formula: {repr(v)}'

print('All checks passed.')
```

If any cell is None or not a formula string, debug and fix before finishing.

## Critical Reminders
- Do NOT open the workbook with data_only=True.
- Do NOT use keep_vba or any macro options.
- Formula strings must start with '='.
- Use PERCENTILE (not PERCENTILE.INC/EXC) to avoid #NAME? errors.
- Do not add sheets, macros, VBA, external links, or helper tabs.
- Preserve all existing formatting.

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