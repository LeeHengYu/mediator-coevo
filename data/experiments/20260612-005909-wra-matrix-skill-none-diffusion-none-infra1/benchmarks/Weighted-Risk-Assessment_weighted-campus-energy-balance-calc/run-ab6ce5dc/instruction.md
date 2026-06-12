# Task Instruction

Execute the following steps to complete the weighted-campus-energy-balance-calc task.

## Phase 1 — Inspect the workbook

1. Open `/root/data/workbook.xlsx` with `openpyxl` (data_only=False).
2. Print the sheet names to confirm `Task` and `Data` exist.
3. On the `Data` sheet, print rows 19–40 (all columns with data) so you can see:
   - The header row (row 20 or 21) that contains years.
   - The column that contains series codes.
   - The data rows 21–38.
4. On the `Task` sheet, print:
   - Row 10 (to see the year headers in columns H–L).
   - Column D, rows 12–31 (to see the series codes for each lookup row).
   - The labels in column A or B for rows 12–17, 19–24, 26–31, 35–40, 42–47, 50 to understand block structure.
   - Any existing content in H12:L50 to confirm the yellow cells are currently empty.

Record the exact row/column layout of the Data sheet (which row is the header row containing years, which column holds series codes, which rows hold the 18 data records). This is critical for building correct INDEX/MATCH formulas.

## Phase 2 — Write lookup formulas in H12:L31

Using the inspection results, construct INDEX/MATCH formulas for every cell in the three blocks:
- H12:L17 (block 1, 6 rows)
- H19:L24 (block 2, 6 rows)
- H26:L31 (block 3, 6 rows)

Formula pattern (adjust references based on your inspection):
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
Replace `$B$21:$Z$38`, `$A$21:$A$38`, `$B$20:$Z$20` with the actual ranges you found during inspection. The key rules:
- The row lookup key is from column D of the current row on Task sheet (use `$D12`, `$D13`, etc. with the column locked).
- The column lookup key is the year from row 10 on Task sheet (use `H$10`, `I$10`, etc. with the row locked).
- The data array, row-key vector, and column-key vector must be consistent and correctly sized.

Write formulas to all 54 cells (18 rows × 5 columns).

## Phase 3 — Net renewable balance (H35:L40)

For each of the 6 campus rows (rows 35–40) and each year column (H–L), write a formula:
```
=(H12 - H19) / H26 * 100
```
Adjust row references so that:
- Row 35 uses the first campus row from block 1 (row 12), block 2 (row 19), block 3 (row 26).
- Row 36 uses rows 13, 20, 27.
- Row 37 uses rows 14, 21, 28.
- Row 38 uses rows 15, 22, 29.
- Row 39 uses rows 16, 23, 30.
- Row 40 uses rows 17, 24, 31.

Verify by checking the campus labels match across blocks.

## Phase 4 — Summary statistics (H42:L47)

For each year column (H–L), write these formulas referencing H35:H40 (adjust column letter per column):
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

Check the row labels in column A/B to confirm which row is which statistic (min, max, median, mean, 25th, 75th). Map accordingly — do NOT assume the order above; use the labels from your inspection.

## Phase 5 — Weighted mean (H50:L50)

For each year column (H–L), write:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the net renewable balance percentages, weighted by baseline energy demand.

## Phase 6 — Save and validate

1. Save the workbook to `/root/output/result.xlsx` (create the output directory if needed).
2. Re-open the saved file and print cells H12, H19, H26, H35, H42, H50 to confirm they contain formula strings (not None).
3. Spot-check that formulas reference the correct Data sheet ranges.

Do NOT add any new sheets, macros, VBA, external links, or helper tabs. Do NOT change existing formatting.

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