# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0 – Preparation
```bash
mkdir -p /root/output
pip install openpyxl --quiet
```

## 1 – Inspect the workbook layout
Open /root/data/workbook.xlsx with openpyxl (data_only=False). Print:
- Sheet names.
- On sheet `Task`: the contents of rows 10-11 (headers/years), column D for rows 12-31 (series codes), column D for rows 35-40 (plant names or series codes), row 41 label, rows 42-47 column D labels, row 50 column D label.
- On sheet `Data`: rows 19-40 to understand the lookup table layout (which row holds what, which column holds what).

This inspection is critical — do NOT skip it. All subsequent formula construction depends on the actual cell contents.

## 2 – Write formulas into the workbook
Using openpyxl (data_only=False, keep_vba=False), load the workbook and write formulas as described below. Use the inspection results to confirm exact row/column references.

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three 6×5 blocks, write an INDEX/MATCH formula that:
- Looks up the series code from column D of the current row against the series-code column on sheet `Data` (rows 21:38).
- Looks up the year from row 10 of the current column against the year row on sheet `Data`.
- Returns the intersection value.

Use the pattern:
```
=INDEX(Data!<data_range>, MATCH(<series_code_cell>, Data!<series_column>, 0), MATCH(<year_cell>, Data!<year_row>, 0))
```
Make sure the `<data_range>`, `<series_column>`, and `<year_row>` references are correct based on the inspection. Use absolute references ($) for the lookup arrays and relative references for the series code cell (column D, same row) and year cell (row 10, same column).

### Step 2a – Net production slack in H35:L40
For each of the 6 rows (plants) and 5 year-columns, write:
```
=(<Finished_Output_cell> - <Scrap_And_Rework_cell>) / <Rated_Production_Capacity_cell> * 100
```
where:
- `Finished Output` values are in H12:L17
- `Scrap And Rework` values are in H19:L24
- `Rated Production Capacity` values are in H26:L31

Confirm the row mapping: row 35 corresponds to row 12, 19, 26 (first plant); row 36 → rows 13, 20, 27; etc.

### Step 2b – Summary statistics in H42:L47
For each column (H through L):
- Row 42: `=MIN(H35:H40)` (adjust column letter)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=_xlfn.PERCENTILE.INC(H35:H40, 0.25)` — use the `_xlfn.` prefix to avoid #NAME? errors in xlsx
- Row 47: `=_xlfn.PERCENTILE.INC(H35:H40, 0.75)`

**Important**: Verify from the inspection which row is MIN, MAX, MEDIAN, MEAN, 25th, 75th by checking the labels in column D for rows 42-47. Adjust the order to match the actual labels.

### Step 3 – Weighted mean in H50:L50
For each column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
(adjust column letter for each of H through L)

## 3 – Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets.

## 4 – Validate
- Reopen /root/output/result.xlsx with openpyxl (data_only=False).
- Print formulas in cells H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50 to confirm they are formulas (not None, not raw values).
- If any cell is None or a raw value, investigate and fix before finishing.
- If a test file exists at /root/test_output.py or similar, run `cd /root && python -m pytest test_output.py -v` and report the results.

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