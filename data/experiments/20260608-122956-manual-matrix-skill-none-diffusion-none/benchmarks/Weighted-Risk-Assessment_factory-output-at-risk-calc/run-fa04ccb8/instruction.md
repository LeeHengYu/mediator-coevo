# Task Instruction

Execute the following steps in a single Python session.

## Phase 0 – Inspect the workbook

```python
import openpyxl, shutil, os

wb = openpyxl.load_workbook('/root/data/workbook.xlsx')

# Inspect 'Data' sheet structure
ds = wb['Data']
print('=== Data sheet dimensions:', ds.dimensions)
for r in range(1, 45):
    row_vals = []
    for c in range(1, 20):
        v = ds.cell(r, c).value
        if v is not None:
            row_vals.append(f"{ds.cell(r,c).coordinate}={v!r}")
    if row_vals:
        print(f"  Row {r}: {row_vals}")

# Inspect 'Task' sheet structure
ts = wb['Task']
print('\n=== Task sheet dimensions:', ts.dimensions)
for r in range(1, 55):
    row_vals = []
    for c in range(1, 20):
        v = ts.cell(r, c).value
        if v is not None:
            row_vals.append(f"{ts.cell(r,c).coordinate}={v!r}")
    if row_vals:
        print(f"  Row {r}: {row_vals}")

wb.close()
```

Carefully read and record:
- The exact column and row positions of series codes and year headers on the **Data** sheet (rows 21-38).
- The series codes in column D on the **Task** sheet for rows 12-17, 19-24, 26-31.
- The years in row 10 of the **Task** sheet (columns H-L).
- The labels/structure of rows 35-40 (Net production slack), rows 42-47 (stats), and row 50 (weighted mean).
- Which rows in the Data sheet correspond to which block (Finished Output, Scrap And Rework, Rated Production Capacity).

## Phase 1 – Write lookup formulas (H12:L31)

Reload the workbook (without data_only) and write INDEX/MATCH formulas into the yellow cells.

For each cell in ranges H12:L17, H19:L24, H26:L31, write a formula like:

```
=INDEX(Data!<value_range>, MATCH($D{row}, Data!<series_code_column>, 0), MATCH({col}$10, Data!<year_header_row>, 0))
```

Where:
- `<value_range>` is the rectangular block of numeric data on the Data sheet (rows 21-38, columns containing the year data).
- `<series_code_column>` is the column on the Data sheet that contains the series codes (must span rows 21-38).
- `<year_header_row>` is the row on the Data sheet that contains the year headers (must span the year columns).
- `$D{row}` uses column-absolute reference so it always looks up from column D of the current Task row.
- `{col}$10` uses row-absolute reference so it always picks the year from row 10.

Adjust the exact Data sheet coordinates based on what Phase 0 reveals. The critical thing is getting the ranges right.

## Phase 2 – Net production slack formulas (H35:L40)

For each of the 6 plants (rows 35-40) and 5 year columns (H-L), write:

```
=({Finished_Output_cell} - {Scrap_And_Rework_cell}) / {Rated_Production_Capacity_cell} * 100
```

Where:
- `{Finished_Output_cell}` is the corresponding cell in H12:L17 (same relative position)
- `{Scrap_And_Rework_cell}` is the corresponding cell in H19:L24
- `{Rated_Production_Capacity_cell}` is the corresponding cell in H26:L31

For example, H35 = `=(H12-H19)/H26*100`, H36 = `=(H13-H20)/H27*100`, etc.

## Phase 3 – Statistical formulas (H42:L47)

For each year column (H-L), in the 6 stat rows:
- Row 42 (MIN): `=MIN(H35:H40)` (adjust column)
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (AVERAGE): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

Verify the stat labels from Phase 0 to confirm the order (min/max/median/mean/25th/75th).

## Phase 4 – Weighted mean (H50:L50)

For each year column (H-L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## Phase 5 – Save

```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
wb.close()
```

## Phase 6 – Verify

Reload the saved file and print the formula strings for a sample of cells (e.g., H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50) to confirm they are formula strings (start with '=') and reference the correct ranges.

## Critical Reminders
- Do NOT use `data_only=True` when loading for writing.
- Do NOT add new sheets, macros, VBA, or external links.
- Do NOT modify any existing formatting.
- All formulas must be Excel formula strings, not Python-computed values.
- Adjust all Data sheet references based on actual inspection in Phase 0. Do not assume coordinates.

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