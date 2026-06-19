# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`:

## 0. Preparation
- `mkdir -p /root/output`
- Open `/root/data/workbook.xlsx` with openpyxl (keep formatting: `load_workbook('/root/data/workbook.xlsx')`).
- Identify the two sheets: `Task` and `Data`. Do NOT create any new sheets.

## 1. Inspect the workbook structure
Before writing any formulas, read and print:
- On sheet `Task`: the contents of column D rows 12–17, 19–24, 26–31 (series codes), and row 10 columns H–L (years). Also print rows 35–40 labels, rows 42–47 labels, and row 50 label.
- On sheet `Data`: rows 21–38 to understand the layout (which row has headers, which column has series codes, which columns have year data). Print row 20 or 21 headers to find the column that holds the series codes and the columns that hold year values.

This inspection is critical — do NOT skip it. Record:
- `data_code_col`: the column letter/number on `Data` where series codes live.
- `data_first_col` / `data_last_col`: the range of year-value columns on `Data`.
- `data_first_row` / `data_last_row`: the row range for the lookup (rows 21–38 or a subset).
- The header row on `Data` that contains the year values matching Task row 10.

## 2. Step 1 — Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write an `INDEX/MATCH` formula that:
- Uses the series code from column D of the same row on `Task`.
- Uses the year from row 10 of the same column on `Task`.
- Looks up the value from the `Data` sheet rows 21:38.

Concrete pattern (adjust column/row references based on inspection):
```
=INDEX(Data!<value_range>, MATCH(D12, Data!<code_column>, 0), MATCH(H10, Data!<year_header_row>, 0))
```
Make sure:
- The `value_range` is a 2D range covering all data rows and year columns on `Data`.
- The `code_column` is a single-column range of series codes on `Data`.
- The `year_header_row` is a single-row range of year headers on `Data`.
- All references use absolute row/column anchoring where needed so each cell resolves correctly.

Write the formula string to each cell using `ws['H12'] = '=INDEX(...)'` etc. Loop over all 3 blocks × 5 columns.

## 3. Step 2 — Net capacity headroom (H35:L40)
For each of the 6 hospital clusters (rows 35–40) and 5 year columns (H–L):
```
=(H12 - H19) / H26 * 100
```
where row 12 = Available Care Slots, row 19 = Occupied Care Slots, row 26 = Staffed Bed Capacity (adjust row offsets per cluster: cluster i uses rows 12+i, 19+i, 26+i for i=0..5).

Verify by inspection that the row mapping is correct (row 35 corresponds to row 12/19/26, row 36 to 13/20/27, etc.).

## 4. Step 2 — Summary statistics (H42:L47)
For each year column (H–L), write formulas in rows 42–47. Check the labels in column D/E of rows 42–47 to determine the order, then assign:
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- MEAN (average): `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)`

**Important**: Match each formula to the actual label in that row. Print the labels first, then assign accordingly.

## 5. Step 3 — Weighted mean (H50:L50)
For each year column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net capacity headroom using Staffed Bed Capacity as weights.

## 6. Save
- Save the workbook to `/root/output/result.xlsx` using `wb.save('/root/output/result.xlsx')`.
- After saving, re-open the file and verify that cells H12, H35, H42, and H50 are not None — they should contain formula strings.
- Print confirmation of the formulas in a few sample cells.

## Key constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting.
- Use openpyxl only.
- Make sure `wb.save()` is called — this was a failure mode in a related task.

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