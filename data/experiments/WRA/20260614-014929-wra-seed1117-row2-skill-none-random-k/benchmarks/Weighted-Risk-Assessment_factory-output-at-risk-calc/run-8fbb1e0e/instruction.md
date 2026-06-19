# Task Instruction

Complete the following steps to update `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`.

## Preliminary Investigation

1. Create the output directory: `mkdir -p /root/output`
2. Install openpyxl if not already available: `pip install openpyxl`
3. Read and thoroughly inspect the workbook structure using a Python script that prints:
   - All sheet names
   - The full contents of the `Task` sheet (all cells from row 1 to ~row 55, columns A through M), printing both values and any existing formulas. Pay special attention to:
     - Column D (series codes) for rows 12-17, 19-24, 26-31, 35-40
     - Row 10 (years) for columns H through L
     - The labels/headers in rows 11, 18, 25, 34, 41, 49
     - Any existing content in cells H42:L47 and H50:L50
   - The full contents of the `Data` sheet rows 1-40 (all populated columns), especially rows 21-38, to understand the data layout (which column has the series code, which row has years, how data is organized)
   - Cell fill colors for the yellow cells mentioned (H12:L17, H19:L24, H26:L31) to confirm they are the target cells

Print everything clearly with row/column labels so the structure is unambiguous.

## Understanding the Layout

Before writing any formulas, determine:
- What series codes appear in column D for each row group (rows 12-17, 19-24, 26-31)
- What years appear in row 10 for columns H-L
- How the Data sheet rows 21-38 are structured: which column contains series codes, which row contains year headers, and where the numeric data lives
- What the three blocks represent (likely Finished Output, Scrap And Rework, Rated Production Capacity based on the Step 2 formula)
- What labels are in rows 35-40 (the six plants for Net production slack)

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Using openpyxl, write Excel formulas (not computed values) into each cell. Use an INDEX/MATCH pattern:
- The lookup should use two inputs: the series code from column D of the current row, and the year from row 10 of the current column.
- The data source is the `Data` sheet rows 21:38.
- Construct the formula so that MATCH finds the series code in the appropriate column of Data!$rows21:38, and MATCH finds the year in the appropriate row, then INDEX retrieves the intersection.

IMPORTANT: After inspecting the Data sheet, determine:
- Which column on Data contains the series codes (the lookup key matching column D on Task)
- Which row on Data contains the year headers (matching row 10 on Task)
- The exact data range for INDEX

The formula pattern will be something like:
`=INDEX(Data!<data_range>, MATCH(D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))`

Adapt the exact references based on what you find in the inspection step. Make sure:
- Row references for the series code column D use relative row (e.g., $D12) so they change per row
- Column references for the year in row 10 use relative column (e.g., H$10) so they change per column
- The Data sheet ranges use absolute references where appropriate

## Step 2: Net production slack in H35:L40 and statistics in H42:L47

For H35:L40, write Excel formulas that compute:
`(Finished Output - Scrap And Rework) / Rated Production Capacity * 100`

Based on the three blocks:
- Finished Output is likely in rows 12-17 (or whichever block the inspection reveals)
- Scrap And Rework is likely in rows 19-24
- Rated Production Capacity is likely in rows 26-31

So for cell H35: `=(H12-H19)/H26*100` (adjust row numbers based on actual inspection; the six plants in rows 35-40 should correspond to the six plants in rows 12-17, 19-24, 26-31 respectively)

For H42:L47, write formulas for column-wise statistics over H35:L40:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

IMPORTANT: Check the labels in column D or nearby columns for rows 42-47 to confirm which row is which statistic. Assign formulas according to the actual labels, not assumed order.

## Step 3: Weighted mean in H50:L50

For each column H through L, write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net production slack percentages (H35:H40) weighted by Rated Production Capacity (H26:H31).

## Saving

Save the workbook to `/root/output/result.xlsx`. Do NOT change any formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.

## Verification

After saving, re-open `/root/output/result.xlsx` with openpyxl and print:
- All formulas in H12:L17 (confirm they are formula strings starting with =)
- All formulas in H19:L24
- All formulas in H26:L31
- All formulas in H35:L40
- All formulas in H42:L47
- All formulas in H50:L50
- Confirm no new sheets were added
- Confirm the formulas use INDEX/MATCH (or one of the allowed lookup patterns) for Step 1
- Confirm SUMPRODUCT is used in Step 3

Also open the workbook in data_only mode and check if any cells that should have formulas are showing None (which would indicate the formula wasn't written correctly).

IMPORTANT NOTES:
- When using openpyxl to write formulas, assign the formula as a string starting with '=' to the cell's value property.
- Make sure to open the workbook without data_only mode when writing formulas.
- The inspection step is critical - do not skip it or assume the layout. Print everything and reason about it before writing any formulas.

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