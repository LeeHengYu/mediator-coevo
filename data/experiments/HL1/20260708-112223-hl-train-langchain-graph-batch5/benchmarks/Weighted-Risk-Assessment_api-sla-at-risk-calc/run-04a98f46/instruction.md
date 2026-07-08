# Task Instruction

You must update the Excel workbook at `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Follow these steps precisely:

## Preliminary Inspection
1. Read the `Task` sheet to understand the layout:
   - Column D contains series codes for rows 12–17 (block 1), 19–24 (block 2), 26–31 (block 3), 35–40 (block 4).
   - Row 10 contains year headers in columns H through L.
   - Rows 35–40 contain six services; rows 42–47 contain summary statistics; row 50 contains the weighted mean.
2. Read the `Data` sheet rows 21–38 to understand the data layout: identify which row contains headers (likely row 21 or the row structure), which column contains series codes, and how years are arranged. Note the exact row/column structure so your MATCH formulas reference the correct ranges.
3. Identify what the three blocks represent:
   - H12:L17 → likely "Latency Budget Preserved" (or similar; read the label in the Task sheet)
   - H19:L24 → likely "Latency Budget Consumed"
   - H26:L31 → likely "Covered Request Capacity"
   Confirm the actual block labels by reading the Task sheet.

## Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write an INDEX/MATCH formula that:
- Uses the series code from column D of the current row (use `$D12` style with absolute column lock)
- Uses the year from row 10 of the current column (use `H$10` style with absolute row lock)
- Looks up the value from the `Data` sheet rows 21:38

The formula pattern should be:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Adjust the exact ranges after inspecting the Data sheet. The `<data_range>` should cover the full numeric area of rows 21:38. The `<series_code_column>` is the column in Data that holds series codes. The `<year_header_row>` is the row in Data that holds year values.

Fill all 6 rows × 5 columns for each of the three blocks (90 formula cells total).

## Step 2: Net SLA Buffer (H35:L40) and Summary Statistics (H42:L47)
For H35:L40, use this formula per cell:
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to the "Latency Budget Preserved" value, H19 to "Latency Budget Consumed", and H26 to "Covered Request Capacity" for the same service and year. Use relative references so the formula adjusts across the 6×5 grid. **Important**: Verify which block is which by reading the Task sheet labels. The row offsets between blocks must be correct (rows 12–17 vs 19–24 vs 26–31 vs 35–40).

For H42:L47, calculate column-wise statistics over H35:L40 (the six Net SLA buffer values per year column):
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

**Check the Task sheet labels in column A/B/C for rows 42–47 to confirm the exact order of statistics.** Assign each formula to the correct row based on the label.

## Step 3: Weighted Mean (H50:L50)
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net SLA buffer percentages weighted by Covered Request Capacity.

## Final Steps
1. Ensure `/root/output/` directory exists (create if needed).
2. Save the workbook to `/root/output/result.xlsx` preserving all existing formatting.
3. Verify the output by reopening it and confirming:
   - Cells H12:L17, H19:L24, H26:L31 contain formulas (not plain values)
   - Cells H35:L40 contain formulas
   - Cells H42:L47 contain formulas
   - Cells H50:L50 contain SUMPRODUCT formulas
   - No extra sheets were added

Use openpyxl for all Excel operations. When opening the workbook, do NOT use `data_only=True` — you need to write and preserve formulas.

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