# Task Instruction

Execute the following steps precisely:

## Phase 1: Inspect the workbook

1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Open `/root/output/result.xlsx` with openpyxl and inspect:
   - **Task sheet**: Print rows 10-50, columns A-L (values and any existing formulas). Pay special attention to:
     - Row 10 (year headers in H10:L10)
     - Column D rows 12-31 (series codes)
     - The structure of H35:L40 (Net capacity headroom), H42:L47 (summary stats labels), H50:L50 (weighted mean)
   - **Data sheet**: Print rows 1-40, focusing on:
     - Which row/column contains the series codes
     - Which row/column contains the year headers
     - The data matrix boundaries (rows 21-38 as mentioned)
     - Exact cell references for the top-left and bottom-right of the data block
3. Print all findings clearly before proceeding.

## Phase 2: Write lookup formulas (Step 1)

Based on the inspection, write INDEX/MATCH formulas in H12:L17, H19:L24, and H26:L31. Use this pattern (adjust ranges based on inspection):

```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Where:
- `<data_range>` is the rectangular block of numeric data on the Data sheet (rows 21:38, columns containing values)
- `<series_code_column>` is the column in Data sheet that holds the series codes, spanning the same rows as the data range
- `<year_header_row>` is the row in Data sheet that holds the year values, spanning the same columns as the data range
- Use `$D12` (absolute column, relative row) and `H$10` (relative column, absolute row) so formulas can be filled across the grid

Apply the formula to all 18 rows × 5 columns = 90 cells in the three blocks.

## Phase 3: Write derived formulas (Step 2)

**Net capacity headroom (H35:L40):**
For each of the 6 hospital clusters (rows 35-40), calculate:
```
=(H12 - H19) / H26 * 100
```
Where H12 corresponds to Available Care Slots, H19 to Occupied Care Slots, H26 to Staffed Bed Capacity. Adjust row references for each cluster row (row 35 uses rows 12,19,26; row 36 uses 13,20,27; etc.).

**Summary statistics (H42:L47):**
- H42: `=MIN(H35:H40)` (minimum)
- H43: `=MAX(H35:H40)` (maximum)
- H44: `=MEDIAN(H35:H40)` (median)
- H45: `=AVERAGE(H35:H40)` (simple mean)
- H46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- H47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

Check the row labels on the Task sheet to confirm the order (min, max, median, mean, p25, p75) matches the actual labels. Adjust if needed.

Fill across columns H through L.

## Phase 4: Write weighted mean formula (Step 3)

In H50:L50, calculate:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
Fill across columns H through L.

## Phase 5: Save and validate

1. Save the workbook to `/root/output/result.xlsx`.
2. Reopen the file and print the formulas in all modified cells to confirm they were written correctly.
3. Verify no extra sheets were added and formatting is preserved.

## Important notes:
- Use `openpyxl` for all operations.
- Do NOT use `data_only=True` when opening for writing.
- When writing formulas, assign them as strings starting with `=`.
- Do NOT add any new sheets, macros, VBA, or external links.
- Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`).
- The inspection phase is critical — do not skip it. Adjust all ranges based on what you actually find in the workbook.

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