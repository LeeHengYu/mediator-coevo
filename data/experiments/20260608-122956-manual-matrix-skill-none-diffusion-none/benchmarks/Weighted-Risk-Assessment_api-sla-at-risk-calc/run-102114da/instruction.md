# Task Instruction

Execute the following steps in a single Python script using openpyxl.

## Phase 0 – Inspection
1. Load `/root/data/workbook.xlsx` with `openpyxl.load_workbook('/root/data/workbook.xlsx')` (do NOT use `data_only=True`).
2. Print the sheet names.
3. On sheet `Task`:
   - Print cells A10:L10 (the header row with years).
   - Print cells A12:G17 (first block labels/series codes), A19:G24 (second block), A26:G31 (third block).
   - Print cells A35:G40 (Net SLA buffer block labels).
   - Print cells A42:G47 (stats block labels).
   - Print cells A50:G50 (Platform SLA Coalition row).
   - Identify which column contains the series/lookup codes (expected column D).
   - Identify which row contains the year headers used in the lookup (expected row 10).
4. On sheet `Data`:
   - Print rows 19 through 40 (columns A through whatever the last populated column is) to see the full data layout.
   - Identify: which column holds the series codes, which row holds the year headers, and the exact data range boundaries.
5. Print a summary of your findings: series-code column on Data sheet, year-header row on Data sheet, top-left and bottom-right of the data body on Data sheet.

## Phase 1 – Lookup Formulas (H12:L31)
Based on the inspection, write INDEX/MATCH formulas into cells H12:L17, H19:L24, and H26:L31 on sheet `Task`.

The formula pattern for each cell should be:
```
=INDEX(Data!<data_body_range>, MATCH($D<row>, Data!<series_code_column_range>, 0), MATCH(<year_col>$<year_header_row>, Data!<year_header_row_range>, 0))
```
Where:
- `$D<row>` is the series code in column D of the current Task row (use absolute column reference `$D`).
- `<year_col>$<year_header_row>` is the year from the header row on Task sheet (use absolute row reference like `H$10`).
- `Data!<data_body_range>` is the rectangular block of numeric values on the Data sheet (rows 21-38, columns after the series code column).
- `Data!<series_code_column_range>` is the series code column on Data sheet for the same rows.
- `Data!<year_header_row_range>` is the year header row on Data sheet spanning the data columns.

IMPORTANT: Construct these references from your Phase 0 inspection. Do NOT guess. Use the actual column letters and row numbers you observed.

Loop over the three blocks (rows 12-17, 19-24, 26-31) and columns H through L, writing the appropriate formula string into each cell.

## Phase 2 – Net SLA Buffer (H35:L40)
For each cell in H35:L40, write a formula:
```
=(<Latency_Budget_Preserved_cell> - <Latency_Budget_Consumed_cell>) / <Covered_Request_Capacity_cell> * 100
```
where:
- The three blocks (rows 12-17, 19-24, 26-31) correspond to three different metrics. Determine from the block labels (printed in Phase 0) which block is "Latency Budget Preserved", which is "Latency Budget Consumed", and which is "Covered Request Capacity".
- The six services in rows 35-40 should correspond positionally to the six services in each block (row 35 ↔ row 12/19/26, row 36 ↔ row 13/20/27, etc.).
- Use cell references, e.g., `=(H12-H19)/H26*100` if block 1=Preserved, block 2=Consumed, block 3=Capacity. Adjust based on actual labels.

## Phase 3 – Summary Statistics (H42:L47)
For each column H through L, write formulas in rows 42-47. Check the labels in column A/B/G of rows 42-47 to determine the order. Expected functions:
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- AVERAGE (simple mean): `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)`

Match each function to the correct row based on the label you read.

## Phase 4 – Weighted Mean (H50:L50)
For each column H through L, write:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net SLA buffer percentages weighted by Covered Request Capacity.

## Phase 5 – Save
1. Save the workbook to `/root/output/result.xlsx` (create the `/root/output/` directory if needed).
2. Re-load the saved file and print cells H12, H19, H26, H35, H42, H50 to confirm they contain formula strings (not None).

## Critical Rules
- Do NOT use `data_only=True` when loading.
- Do NOT add sheets, macros, VBA, external links, or helper columns.
- Do NOT modify any existing formatting.
- Do NOT modify any cells outside the specified yellow ranges.
- Use `os.makedirs('/root/output', exist_ok=True)` before saving.

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