# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl and inspect:
- Sheet names (confirm `Task` and `Data` exist)
- On `Task` sheet: read row 10 (years in H10:L10), column D rows 12-17, 19-24, 26-31 (series codes), row 35-40 labels/column D, rows 42-47 labels, row 50 label
- On `Data` sheet: read rows 21-38 to understand the data layout (what's in each column/row, where series codes are, where years are as column headers)
- Print all of this so we understand the exact layout before writing any formulas

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a spreadsheet formula (not a Python computation) using INDEX/MATCH. The formula pattern should be:

`=INDEX(Data!<data_range>, MATCH(<series_code_ref>, Data!<series_code_column>, 0), MATCH(<year_ref>, Data!<year_header_row>, 0))`

Where:
- `<series_code_ref>` is the cell reference to column D of the current row on sheet `Task` (e.g., `$D12` for row 12)
- `<year_ref>` is the cell reference to the year in row 10 (e.g., `H$10` for column H)
- `<data_range>` is the rectangular block on `Data` sheet covering rows 21-38 and the columns that contain the numeric data
- `<series_code_column>` is the column on `Data` that holds the series codes (same rows 21-38)
- `<year_header_row>` is the row on `Data` that holds the year headers (same columns as data)

IMPORTANT: After inspecting the Data sheet layout, determine the exact ranges. The series codes might be in column A or B of Data, and year headers might be in a specific row. Adjust the formula accordingly. Use absolute references where needed (lock the row for year with `$10`, lock the column for series code with `$D`).

Use openpyxl to write these formulas as strings into each cell. Make sure to set `workbook.calculation.calcMode` or similar if needed, but primarily just write the formula strings.

## 3. Calculate Net container flow in H35:L40

For each cell in H35:L40, write a formula:
`=(<loaded_inbound_cell> - <loaded_outbound_cell>) / <terminal_throughput_cell> * 100`

Where:
- `<loaded_inbound_cell>` is the corresponding cell from H12:L17 (Loaded Containers Inbound block)
- `<loaded_outbound_cell>` is the corresponding cell from H19:L24 (Loaded Containers Outbound block)
- `<terminal_throughput_cell>` is the corresponding cell from H26:L31 (Terminal Throughput Capacity block)

For example, H35 = `=(H12-H19)/H26*100`, H36 = `=(H13-H20)/H27*100`, etc.

BUT FIRST verify which block is which by checking the labels/series codes. The three blocks (rows 12-17, 19-24, 26-31) correspond to three different metrics. Map them correctly to: Loaded Containers Inbound, Loaded Containers Outbound, and Terminal Throughput Capacity. Similarly verify that rows 35-40 correspond to the same six ports in the same order.

## 4. Summary statistics in H42:L47

For each column (H through L), write formulas in rows 42-47. Check the labels in column D (or wherever) for rows 42-47 to determine which statistic goes where. Expected statistics: minimum, maximum, median, simple mean, 25th percentile, 75th percentile.

Use these Excel functions:
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- AVERAGE: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40, 0.25)` (or PERCENTILE.INC)
- 75th percentile: `=PERCENTILE(H35:H40, 0.75)` (or PERCENTILE.INC)

Match each formula to the correct row based on the label.

## 5. Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net container flow percentages using Terminal Throughput Capacity as weights.

## 6. Save

Save the workbook to `/root/output/result.xlsx`. Do NOT create new sheets. Do NOT change formatting. Use `openpyxl` with `keep_vba=False` (default). When loading, do NOT use `data_only=True` (we need to preserve formulas).

## 7. Verify

Reload `/root/output/result.xlsx` and print the formula content of representative cells (e.g., H12, L17, H35, H40, H42, H47, H50, L50) to confirm formulas were written correctly.

## Critical Notes
- Write FORMULA STRINGS, not computed values. Each cell must contain an Excel formula starting with `=`.
- Use `$` signs appropriately: lock row 10 reference with `$10` and lock column D with `$D` in lookup formulas.
- Do not add or remove any sheets.
- Do not modify any cells outside the specified ranges.
- Preserve all existing formatting (do not call any formatting methods).
- The inspection in step 1 is critical — do not skip it. The exact row/column layout of the Data sheet determines the correctness of every formula.

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