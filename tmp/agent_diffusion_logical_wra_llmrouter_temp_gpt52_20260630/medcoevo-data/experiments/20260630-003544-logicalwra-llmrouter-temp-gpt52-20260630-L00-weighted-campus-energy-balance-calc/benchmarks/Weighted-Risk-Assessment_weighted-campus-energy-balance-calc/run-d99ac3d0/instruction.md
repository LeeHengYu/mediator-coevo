# Task Instruction

Execute the following steps precisely to complete the weighted campus energy balance workbook task.

## Pre-work: Inspect the workbook

1. Copy the source workbook:
   ```
   cp /root/data/workbook.xlsx /root/output/result.xlsx
   ```
2. Use `openpyxl` to inspect the workbook structure. Print:
   - Sheet names
   - The contents of sheet `Task` rows 1–55, columns A–M (both `.value` and any existing formulas). Pay special attention to:
     - Row 10 (years in H10:L10)
     - Column D rows 12–17, 19–24, 26–31 (series codes)
     - The labels in column A or B for rows 12–17, 19–24, 26–31 (to understand which block is Renewable Generation, Grid Consumption, Baseline Energy Demand)
     - Rows 35–40 (campus names/labels for Net renewable balance)
     - Rows 42–47 (labels: min, max, median, mean, 25th percentile, 75th percentile)
     - Row 50 (MCEC weighted mean)
   - The contents of sheet `Data` rows 21–38, all columns that have data. Identify:
     - Which column contains the series codes (lookup keys)
     - Which row contains years (column headers)
     - The data layout so you know whether VLOOKUP/HLOOKUP/INDEX-MATCH is appropriate

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write formulas into the yellow cells. For each cell at position (row, col) in these ranges:
- The series code is in column D of that row (e.g., `$D12` for row 12, with the column locked so it doesn't shift when copied across)
- The year is in row 10 of that column (e.g., `H$10` for column H, with the row locked)
- The data source is on sheet `Data` rows 21:38

Use `INDEX(MATCH, MATCH)` pattern. Determine the exact data range on the `Data` sheet from your inspection. The formula pattern should be something like:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```
Adjust the ranges based on what you actually find in the Data sheet. Make sure:
- The series code column reference is absolute in the column ($D)
- The year row reference is absolute in the row ($10)
- This allows the formula to be consistent across the entire block

Use `openpyxl` to write these formulas as strings (not computed values). Open with `data_only=False` (default).

**Important**: When writing formulas with openpyxl, the formula string must start with `=` and use the exact Excel syntax. Use comma as the argument separator (not semicolon).

## Step 2: Net renewable balance in H35:L40 and statistics in H42:L47

For H35:L40, each row corresponds to a campus. Based on the block structure:
- Rows 12–17: one metric (identify which from inspection)
- Rows 19–24: another metric
- Rows 26–31: another metric

The formula is: `(Renewable Generation - Grid Consumption) / Baseline Energy Demand * 100`

Map the correct row blocks to these three quantities. For example, if rows 12–17 are Renewable Generation, rows 19–24 are Grid Consumption, and rows 26–31 are Baseline Energy Demand, then for cell H35:
```
=(H12-H19)/H26*100
```
Adjust row references so that row 35 maps to the first campus (row 12, 19, 26), row 36 maps to the second (row 13, 20, 27), etc.

For H42:L47 (column-wise statistics over H35:L40):
- Row 42 (min): `=MIN(H35:H40)`
- Row 43 (max): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**Check the actual labels in column A/B/C for rows 42–47 to confirm which row gets which function.** Match the function to the label.

## Step 3: Weighted mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean of the Net renewable balance percentages using Baseline Energy Demand as weights.

## Final steps

1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open and verify:
   - Cells H12:L17, H19:L24, H26:L31 contain formula strings (not None, not plain values)
   - Cells H35:L40 contain formula strings
   - Cells H42:L47 contain formula strings
   - Cell H50:L50 contains formula strings
   - No new sheets were added
   - Print a sample of formulas from each block to confirm correctness

## Critical constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs
- Do NOT change existing formatting
- Use `openpyxl` with default mode (not data_only) so formulas are preserved
- Ensure `/root/output/` directory exists before saving

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