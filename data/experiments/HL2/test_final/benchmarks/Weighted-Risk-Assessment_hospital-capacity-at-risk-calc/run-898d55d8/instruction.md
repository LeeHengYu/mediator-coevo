# Task Instruction

Execute the following steps exactly in order.

## 0 – Inspect the workbook

```python
import openpyxl, json
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for name in wb.sheetnames:
    print(f'--- Sheet: {name} ---')
    ws = wb[name]
    print(f'Dimensions: {ws.dimensions}')
    # Print first 50 rows, columns A-M
    for row in ws.iter_rows(min_row=1, max_row=55, max_col=13, values_only=False):
        print([(c.coordinate, c.value) for c in row])
wb.close()
```

Record:
- The exact series codes in Task!D12:D17, D19:D24, D26:D31 (the three blocks).
- The years in Task!H10:L10.
- The layout of Data sheet rows 21-38: which row has which series code, which column has which year, and whether the data is arranged with series codes in a column and years across columns (or transposed).
- The labels in Task!C35:C40 (Net capacity headroom clusters), C42:C47 (summary stats), C50 (weighted mean).
- Which cells are currently empty (the yellow cells to fill).

## 1 – Build the formulas with openpyxl

Open the workbook (with `data_only=False` so formulas are preserved) and write formulas.

### 1a – Lookup formulas (H12:L17, H19:L24, H26:L31)

For each of the three 6×5 blocks, write an INDEX-MATCH formula in every cell.

The pattern for cell at row `r`, column `c` (where `c` maps to H=8, I=9, … L=12):

```
=INDEX(Data!<year_data_range>, MATCH($D{r}, Data!<series_code_column>, 0), MATCH(<year_cell>, Data!<year_header_row>, 0))
```

Determine the exact ranges from your inspection:
- `<series_code_column>` – the column on Data sheet that holds series codes for rows 21-38.
- `<year_header_row>` – the row on Data sheet that holds the year headers.
- `<year_data_range>` – the rectangular block of numeric data on Data sheet (rows 21-38, year columns).

Make the series-code reference use `$D{r}` (column-absolute) and the year reference use `{col_letter}$10` (row-absolute) so the formula anchors correctly.

IMPORTANT: Use `$` signs carefully. The Data ranges should be fully absolute (e.g., `Data!$B$21:$B$38` for series codes).

### 1b – Net capacity headroom (H35:L40)

For each of the 6 rows and 5 year-columns, write:
```
=(<Available_cell> - <Occupied_cell>) / <Staffed_cell> * 100
```
where:
- Available Care Slots = block H12:L17 (row 12+i for cluster i=0..5)
- Occupied Care Slots = block H19:L24 (row 19+i)
- Staffed Bed Capacity = block H26:L31 (row 26+i)

So for row 35, col H: `=(H12-H19)/H26*100`
For row 36, col H: `=(H13-H20)/H27*100`
… and so on for all 30 cells.

### 1c – Summary statistics (H42:L47)

For each year-column (H through L):
- Row 42 (Min):    `=MIN(H35:H40)`
- Row 43 (Max):    `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean):   `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Use `PERCENTILE` — NOT `PERCENTILE.INC` or `PERCENTILE.EXC`. The previous two iterations failed with #NAME? because a dotted variant was used. Use the plain `PERCENTILE` function.

### 1d – Weighted mean (H50:L50)

For each year-column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## 2 – Save

Save the workbook to `/root/output/result.xlsx`. Create the output directory if needed.

## 3 – Validate

Reload the saved workbook and print all formula cells in the Task sheet to confirm:
- Formulas are present (not values).
- No obvious typos.
- `PERCENTILE` (not `PERCENTILE.INC`) is used in rows 46-47.
- All 30 lookup cells, 30 headroom cells, 30 summary cells, and 5 weighted-mean cells are populated.
- No extra sheets were added.

Print the sheet names and a count of formula cells as final confirmation.

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