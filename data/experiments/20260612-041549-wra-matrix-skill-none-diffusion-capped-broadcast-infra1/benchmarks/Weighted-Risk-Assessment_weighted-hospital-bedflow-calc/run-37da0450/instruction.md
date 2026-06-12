# Task Instruction

Execute the following steps exactly in order.

## Step 0 — Inspect the workbook structure

```python
import openpyxl, pprint
wb = openpyxl.load_workbook('/root/data/workbook.xlsx', data_only=True)

# Task sheet layout
ts = wb['Task']
print('=== Task sheet, rows 1-55, cols A-M ===')
for row in ts.iter_rows(min_row=1, max_row=55, min_col=1, max_col=13, values_only=False):
    print([(c.coordinate, c.value) for c in row])

# Data sheet layout
ds = wb['Data']
print('\n=== Data sheet, rows 1-45, cols A-Z ===')
for row in ds.iter_rows(min_row=1, max_row=45, min_col=1, max_col=26, values_only=False):
    print([(c.coordinate, c.value) for c in row])

wb.close()
```

Study the output carefully. Identify:
- Row 10 on Task: which cells H10:L10 hold the year headers (e.g. 2019, 2020, …).
- Column D on Task: which cells D12:D17, D19:D24, D26:D31 hold the series codes.
- Data sheet rows 21-38: the exact layout — which row holds headers (series codes or years), which column holds what.
- The structure that tells you whether series codes run down a column or across a row in Data!21:38.

## Step 1 — Write lookup formulas in H12:L31

Open the workbook WITHOUT data_only (so formulas are preserved):

```python
import openpyxl
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
ts = wb['Task']
```

Based on the inspection, construct INDEX/MATCH formulas. The general pattern for cell (r, c) is:

```
=INDEX(Data!<value_range>, MATCH(<series_code_cell>, Data!<series_code_column>, 0), MATCH(<year_cell>, Data!<year_row>, 0))
```

Where:
- `<series_code_cell>` is e.g. `$D12` (lock column with $D)
- `<year_cell>` is e.g. `H$10` (lock row with $10)
- `<value_range>` is the rectangular data block on the Data sheet (rows 21-38, the columns that contain numeric values)
- `<series_code_column>` is the column in Data that contains the series codes (same rows as value_range)
- `<year_row>` is the row in Data that contains the year headers (same columns as value_range)

IMPORTANT: Use absolute references for the Data ranges (e.g. Data!$B$21:$F$38) so they don't shift. Use mixed references for the lookup keys ($D12 locks column, H$10 locks row).

Write formulas into every cell in H12:L17, H19:L24, H26:L31 (3 blocks × 6 rows × 5 columns = 90 cells).

After writing, re-read a few cells to confirm the formula string is stored.

## Step 2 — Net patient flow in H35:L40

For each hospital row i (i = 0..5):
- Patient Admissions are in rows 12-17 (H12:L17)
- Patient Discharges are in rows 19-24 (H19:L24)
- Effective Bed Capacity are in rows 26-31 (H26:L31)

The formula for cell H35 (first hospital, first year) is:
```
=(H12-H19)/H26*100
```
Generalize with appropriate row offsets for all 6 hospitals across 5 years.

Write these into H35:L40.

## Step 3 — Summary statistics in H42:L47

For each year column (H through L), compute column-wise stats over H35:L40:
- Row 42: MIN, e.g. `=MIN(H35:H40)`
- Row 43: MAX, e.g. `=MAX(H35:H40)`
- Row 44: MEDIAN, e.g. `=MEDIAN(H35:H40)`
- Row 45: AVERAGE, e.g. `=AVERAGE(H35:H40)`
- Row 46: PERCENTILE, e.g. `=PERCENTILE(H35:H40,0.25)` — NOTE: use `PERCENTILE` not `PERCENTILE.INC` to avoid #NAME? errors in some engines
- Row 47: PERCENTILE, e.g. `=PERCENTILE(H35:H40,0.75)`

CHECK the labels in column A/B/C/D for rows 42-47 to confirm which row is which statistic. Adjust the row assignments if the labels differ from the order above.

IMPORTANT: Use `PERCENTILE(range, k)` — do NOT use `PERCENTILE.INC` or `PERCENTILE.EXC` as these can cause #NAME? errors. Similarly use `MEDIAN`, `MIN`, `MAX`, `AVERAGE` (all standard functions).

## Step 4 — Weighted mean in H50:L50

For each year column col (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

This computes the weighted mean of Net patient flow using Effective Bed Capacity as weights.

Write into H50:L50.

## Step 5 — Save and verify

```python
import os
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
wb.close()
```

Then reload and verify:
```python
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
ts2 = wb2['Task']
# Check formulas exist
for cell_addr in ['H12', 'L17', 'H19', 'L24', 'H26', 'L31', 'H35', 'L40', 'H42', 'L47', 'H50', 'L50']:
    print(f"{cell_addr}: {ts2[cell_addr].value}")
wb2.close()
```

Every checked cell must show a formula string (starting with '='). If any cell shows None, diagnose and fix before finishing.

## Critical Reminders
- Do NOT use data_only=True when loading the workbook for editing — that strips formulas.
- Do NOT add new sheets, macros, or VBA.
- Do NOT change formatting.
- Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC`.
- Verify the exact Data sheet layout before constructing formulas — do not assume column/row positions.
- The previous failure was because cells were empty (None) in the output. Ensure every target cell has a formula written to it and the file is saved to /root/output/result.xlsx.

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