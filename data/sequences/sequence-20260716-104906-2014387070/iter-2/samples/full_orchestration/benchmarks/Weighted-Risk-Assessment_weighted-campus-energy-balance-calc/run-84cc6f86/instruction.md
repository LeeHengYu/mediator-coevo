# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 – Preliminary inspection
1. Copy the workbook: `cp /root/data/workbook.xlsx /root/output/result.xlsx`.
2. Open `/root/output/result.xlsx` with openpyxl (data_only=False) and inspect:
   - Sheet names (confirm `Task` and `Data` exist).
   - On `Task`: print rows 10-50 for columns D through L so you can see the series codes in column D, the years in row 10, and the layout of the yellow target cells.
   - On `Data`: print rows 21-38 (all columns) to understand the lookup source table structure (which column holds series codes, which row holds years, where values start).
   - Print `Task` rows 26-31 columns D-L (Baseline Energy Demand block) and rows 35-40 columns D-L to see campus names / series codes for Net renewable balance.
   - Print `Task` rows 42-47 column D (stat labels) and row 50 column D (weighted-mean label).
3. Record exact cell references: where series codes live on Data (likely column A or B of rows 21-38), where year headers live on Data (likely some row above 21, or row 21 itself – check rows 1-21), and the data value area.

## 1 – Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For every cell in those three blocks, write an Excel formula that:
- Uses the series code from column D of that row on `Task`.
- Uses the year from row 10 of that column on `Task`.
- Looks up the value from `Data!` rows 21:38.

Use INDEX/MATCH (safest cross-sheet pattern):
```
=INDEX(Data!<value_area>, MATCH(Task!$D<row>, Data!<series_code_column>, 0), MATCH(Task!<col>$10, Data!<year_row>, 0))
```
Adjust the exact ranges after inspection. Lock the series-code column reference with `$D` and the year row with `$10` so the formula can be dragged across columns and down rows correctly.

Write these formulas with openpyxl by assigning the formula string (starting with `=`) to each cell. Do NOT use `data_only=True` when writing.

## 2 – Step 2a: Net renewable balance (H35:L40)
For each campus row (35-40) and each year column (H-L), write:
```
=(<Renewable_Generation_cell> - <Grid_Consumption_cell>) / <Baseline_Energy_Demand_cell> * 100
```
where the three referenced cells come from the corresponding campus row in the three blocks filled in Step 1 (H12:L17 = one metric, H19:L24 = another, H26:L31 = another). Determine which block is Renewable Generation, which is Grid Consumption, and which is Baseline Energy Demand by reading the labels in column D (or nearby) during inspection.

## 3 – Step 2b: Summary statistics (H42:L47)
For each year column (H-L), write column-wise formulas over H35:H40 (etc.):
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (AVERAGE): `=AVERAGE(H35:H40)`
- Row 46 (25th pctl): `=PERCENTILE(H35:H40,0.25)`  (or PERCENTILE.INC)
- Row 47 (75th pctl): `=PERCENTILE(H35:H40,0.75)`  (or PERCENTILE.INC)
Match the stat label in column D of each row to the correct function. If the labels are in a different order, follow the labels.

## 4 – Step 3: Weighted mean (H50:L50)
For each year column write:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the Net renewable balance percentages as values and Baseline Energy Demand as weights.

## 5 – Save and verify
1. Save the workbook (`wb.save('/root/output/result.xlsx')`).
2. Reopen it and print all formula cells to confirm they are formula strings (not None or numeric).
3. Spot-check a few formulas for correct cell references.
4. Confirm no new sheets were added and sheet names are unchanged.
5. Confirm the file exists at `/root/output/result.xlsx`.

## Key cautions
- Do NOT use `data_only=True` when opening for writing; that strips formulas.
- Preserve all existing formatting, merged cells, styles. Only assign `.value` to the target cells.
- Use `PERCENTILE.INC` if the workbook already uses that variant; otherwise plain `PERCENTILE` is fine.
- Double-check during inspection whether Data rows 21-38 include a header row or are pure data rows; adjust INDEX range accordingly.
- If the year headers on Data are in a row (e.g., row 20 or row 21), use that exact row in the MATCH for columns.

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