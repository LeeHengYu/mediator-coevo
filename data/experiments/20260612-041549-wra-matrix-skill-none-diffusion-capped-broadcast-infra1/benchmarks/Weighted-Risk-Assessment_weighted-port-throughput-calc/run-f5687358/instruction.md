# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
- Open `/root/data/workbook.xlsx` with openpyxl (keep formatting: load with `data_only=False`).
- Print the sheet names to confirm `Task` and `Data` exist.
- On sheet `Task`:
  - Print rows 10-50, columns D through L, to understand the layout: what's in D12:D17, D19:D24, D26:D31 (series codes), what's in H10:L10 (years), what's in rows 35-40 (port names/structure), rows 42-47 (stat labels), and row 50 (CPA weighted mean).
- On sheet `Data`:
  - Print rows 21-38 fully (all columns) to understand the lookup source structure. Identify which column holds the series codes, which row holds years, and how the data is arranged (is it a vertical table with series codes in one column and years across columns, or something else?).
- Print this information before writing any formulas.

## 2. Determine the lookup pattern
Based on the inspection:
- Identify the exact column on `Data` that contains series codes (let's call it the key column).
- Identify the row on `Data` that contains years (the header row for the data range).
- Determine the full data range on `Data` rows 21:38 to use in lookups.
- Choose INDEX/MATCH as the lookup pattern since it's the most flexible.

## 3. Populate H12:L17, H19:L24, H26:L31 with lookup formulas
For each cell in these ranges, write a formula like:
```
=INDEX(Data!<data_range>, MATCH($D<row>, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```
where:
- `$D<row>` is the series code in column D of the current row (absolute column reference)
- `H$10` (or I$10, J$10, etc.) is the year in row 10 (absolute row reference)
- `<data_range>` is the rectangular block on Data rows 21:38 containing the numeric values
- `<series_code_column>` is the column range containing series codes
- `<year_row>` is the row range containing years

IMPORTANT: Use openpyxl to write these as string formulas (e.g., `cell.value = '=INDEX(...)'`). Make sure to use proper Excel references. Use absolute references where needed ($D for column D, $10 for row 10).

## 4. Populate H35:L40 with Net Container Flow formulas
Based on the layout:
- H12:L17 should be Loaded Containers Inbound (first block)
- H19:L24 should be Loaded Containers Outbound (second block)  
- H26:L31 should be Terminal Throughput Capacity (third block)

Verify this by checking labels near rows 11, 18, 25 on the Task sheet.

For each cell in H35:L40, the formula is:
```
=(H12-H19)/H26*100
```
(adjusting row references for each port, and column for each year). For example:
- H35 = `=(H12-H19)/H26*100`
- H36 = `=(H13-H20)/H27*100`
- H40 = `=(H17-H24)/H31*100`
- I35 = `=(I12-I19)/I26*100`
etc.

## 5. Populate H42:L47 with summary statistics
For each column (H through L), in rows 42-47, write formulas. Check the labels in column D or nearby for rows 42-47 to determine the exact order of: minimum, maximum, median, simple mean, 25th percentile, 75th percentile. Then write:
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- MEAN: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)`

Match each formula to the correct row based on the labels found.

## 6. Populate H50:L50 with weighted mean
For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net Container Flow percentages using Terminal Throughput Capacity as weights.

## 7. Save
Save the workbook to `/root/output/result.xlsx`. Do NOT use `data_only=True` when loading. Preserve all existing formatting.

## 8. Verify
- Reopen the saved file and print the formula cells to confirm formulas are written correctly.
- Confirm no extra sheets were added.
- Confirm the file opens without errors.

## Critical Notes
- Do NOT use `data_only=True` when loading — this would strip formulas.
- Write formulas as strings starting with `=`.
- Do not modify any cells outside the specified ranges.
- Do not change formatting, add sheets, or add macros.
- Inspect the actual workbook structure FIRST before writing any formulas — the exact column letters and row numbers on the Data sheet must be determined from inspection, not assumed.

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