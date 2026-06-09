# Task Instruction

Execute the following steps in order.

## Step 0 – Inspect the workbook

```python
import openpyxl, os, shutil

wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
print('Sheet names:', wb.sheetnames)

ws_task = wb['Task']
ws_data = wb['Data']

# Print Task sheet layout: rows 1-55, columns A-M
print('=== Task sheet ===')
for row in ws_task.iter_rows(min_row=1, max_row=55, min_col=1, max_col=13, values_only=False):
    for cell in row:
        if cell.value is not None:
            print(f"  {cell.coordinate}: {repr(cell.value)}")

# Print Data sheet layout: rows 1-45, columns A-Z (to find the matrix)
print('=== Data sheet ===')
for row in ws_data.iter_rows(min_row=1, max_row=45, min_col=1, max_col=26, values_only=False):
    for cell in row:
        if cell.value is not None:
            print(f"  {cell.coordinate}: {repr(cell.value)}")

wb.close()
```

Study the output carefully. Identify:
- The series codes in Task!D12:D17, D19:D24, D26:D31 (the lookup keys).
- The years in Task!H10:L10 (the column keys).
- The layout of Data!rows 21:38 — which column holds the series code, which row holds the year headers, and where the numeric values start.
- The hospital names in the Net patient flow block (rows 35-40) and the labels in rows 42-47.
- The weighted-mean row 50.

## Step 1 – Write lookup formulas in H12:L17, H19:L24, H26:L31

Based on the cross-task artifact, use INDEX/MATCH. The pattern for each cell is:

```
=INDEX(Data!<value_range>, MATCH(Task!$D<row>, Data!<series_code_column>, 0), MATCH(Task!H$10, Data!<year_header_row>, 0))
```

Concretely, after inspecting the Data sheet:
- Determine the exact range of the data matrix (e.g., Data!B22:F38 or similar).
- Determine the series-code column (e.g., Data!A22:A38).
- Determine the year-header row (e.g., Data!B21:F21).
- Construct the INDEX/MATCH formula with appropriate absolute references so that:
  - The series code reference locks the column ($D) but uses the current row.
  - The year reference locks the row ($10) but uses the current column.

Write these formulas using openpyxl:
```python
ws_task.cell(row=r, column=c).value = formula_string
```

Do this for all three blocks (H12:L17, H19:L24, H26:L31). Each block has 6 rows × 5 columns = 30 cells, totaling 90 formula cells.

## Step 2 – Net patient flow formulas in H35:L40

For each hospital row (35-40) and each year column (H-L, i.e., columns 8-12):
```
=(H12 - H19) / H26 * 100
```
where the row offsets correspond to the same hospital across the three blocks:
- Row 35 uses rows 12, 19, 26
- Row 36 uses rows 13, 20, 27
- Row 37 uses rows 14, 21, 28
- Row 38 uses rows 15, 22, 29
- Row 39 uses rows 16, 23, 30
- Row 40 uses rows 17, 24, 31

Write these formulas.

## Step 2b – Summary statistics in H42:L47

For each year column (H-L):
- Row 42 (Minimum): `=MIN(H35:H40)`
- Row 43 (Maximum): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

Use the legacy `PERCENTILE` function (not PERCENTILE.INC).

## Step 3 – Weighted mean in H50:L50

For each year column (H-L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

This computes the weighted mean of net patient flow percentages using effective bed capacity as weights.

## Step 4 – Save

```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
wb.close()
```

## Step 5 – Verify

Reopen `/root/output/result.xlsx` with openpyxl and print the values/formulas in cells H12, H19, H26, H35, H42, H50 to confirm they contain formula strings (starting with '='). Also confirm the file exists and has a reasonable size.

If any test script exists (e.g., `/root/test_output.py` or similar), run it with `pytest` to check.

## IMPORTANT NOTES
- Do NOT use `data_only=True` when loading the workbook for writing.
- Make sure all formula strings start with '=' and use correct Excel syntax.
- Use `Data!` prefix for all references to the Data sheet within formulas.
- Lock references appropriately: $D for series code column, $10 for year row.
- Do not modify any existing formatting, do not add sheets or macros.
- Adapt all row/column references based on what you actually see in Step 0. The numbers above are estimates — the inspection output is the ground truth.

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