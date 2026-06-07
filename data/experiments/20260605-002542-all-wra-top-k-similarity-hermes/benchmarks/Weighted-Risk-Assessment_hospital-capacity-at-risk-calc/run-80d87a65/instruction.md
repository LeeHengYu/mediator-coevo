# Task Instruction

Complete the following task to populate formulas in an Excel workbook.

## Overview
Update `/root/data/workbook.xlsx` by adding spreadsheet formulas to the `Task` sheet. Save the result to `/root/output/result.xlsx`. Do NOT add sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

## Step 0: Inspect the workbook thoroughly
1. Create the output directory: `mkdir -p /root/output`
2. Use `openpyxl` (Python) to open `/root/data/workbook.xlsx` and inspect BOTH sheets:
   - On `Task` sheet: print rows 1-55 (all columns A through at least M). Pay special attention to:
     - Column D (series codes) for rows 12-17, 19-24, 26-31
     - Row 10 (years in columns H through L)
     - The labels/headers in rows 11, 18, 25, 34, 41, 49
     - Any existing formulas or values in the target ranges
     - The exact text in cells around H35:L40, H42:L47, H50:L50
   - On `Data` sheet: print rows 1-40 (all columns). Pay special attention to:
     - Rows 21-38: understand the data layout (what's in each column/row)
     - Identify how series codes and years map to the data
     - Determine whether data is arranged with series codes in rows or columns
3. Print the exact content so you can see the structure before writing any formulas.

## Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For each yellow cell in these three blocks:
- The formula must use TWO inputs: the series code from column D of that row AND the year from row 10 of that column.
- The source data is on `Data` sheet rows 21:38.
- Use one of these patterns: INDEX+MATCH, VLOOKUP+MATCH, HLOOKUP+MATCH, or XLOOKUP+MATCH.
- Choose the pattern that fits the data layout you discovered in Step 0.
- Make sure references are appropriately anchored (mixed/absolute references) so formulas can be filled across the range correctly. Column D references should lock the column; row 10 references should lock the row.
- IMPORTANT: Inspect the Data sheet layout carefully. If series codes are in a column, INDEX+MATCH or VLOOKUP+MATCH is natural. If series codes are in a row, HLOOKUP+MATCH or INDEX with two MATCH calls may be needed.

## Step 2: Net capacity headroom in H35:L40 and statistics in H42:L47
- H35:L40: For each of the six hospital clusters, calculate:
  `(Available Care Slots - Occupied Care Slots) / Staffed Bed Capacity * 100`
  Identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to Available Care Slots, Occupied Care Slots, and Staffed Bed Capacity by reading the block headers/labels.
- H42:L47: For each column (year), calculate these six statistics over the H35:L40 range:
  - Minimum: `=MIN(H35:H40)` (adjust column)
  - Maximum: `=MAX(H35:H40)`
  - Median: `=MEDIAN(H35:H40)`
  - Simple mean: `=AVERAGE(H35:H40)`
  - 25th percentile: Use `PERCENTILE` or `PERCENTILE.INC` with 0.25
  - 75th percentile: Use `PERCENTILE` or `PERCENTILE.INC` with 0.75
  Match each statistic to the correct row based on the labels in column A-G.

## Step 3: Weighted mean in H50:L50
- Use `SUMPRODUCT` to calculate the weighted mean of the Step 2 percentages (H35:H40) weighted by Staffed Bed Capacity (H26:H31):
  `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)` (adjust columns for each)
- Verify this is in the row labeled for `Regional Care Grid`.

## Step 4: Save and validate
1. Save to `/root/output/result.xlsx` using openpyxl, preserving all formatting.
2. Re-open the saved file and print the formula cells (H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50) to verify formulas were written correctly.
3. Check that no extra sheets were added and existing sheets are intact.

## Critical Notes
- Write Excel formulas as strings (e.g., cell.value = '=INDEX(...)'), NOT computed Python values.
- When opening with openpyxl, do NOT use data_only=True for the write pass.
- Preserve existing cell formatting: do not clear styles, fills, fonts, or borders.
- The statistics rows (H42:L47) must match the labels in the Task sheet - inspect what label is in each row before assigning formulas.
- Use `Translator` or careful absolute/relative references when filling formulas across a range.

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