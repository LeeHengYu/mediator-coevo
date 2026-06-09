# Task Instruction

Execute the following steps carefully in order.

## Phase 0 – Inspect the workbook

1. Open `/root/data/workbook.xlsx` with openpyxl (do NOT use `data_only=True`).
2. Print the sheet names to confirm `Task` and `Data` exist.
3. On the `Data` sheet:
   - Print rows 20–40 (all columns up to column N or so) so you can see the exact layout: which column holds the series codes, which row holds the year headers, and where the numeric data lives.
   - Identify: (a) the column that contains the series/indicator codes (likely column A or B), (b) the row that contains the year values, (c) the top-left and bottom-right cell of the data block spanning rows 21–38.
4. On the `Task` sheet:
   - Print rows 10–50 (columns D through L) so you can see: the year row (row 10), the series codes in column D for rows 12–17, 19–24, 26–31, the labels for rows 35–40 (regions), rows 42–47 (statistics), and row 50.
   - Print cells H12:L17, H19:L24, H26:L31 to confirm they are currently empty (the yellow cells).
   - Print H35:L40, H42:L47, H50:L50 to confirm they are currently empty.

Record all findings before proceeding. Do NOT write any formulas until you have confirmed the exact layout.

## Phase 1 – Build lookup formulas for H12:L31

Based on the inspection, construct INDEX/MATCH formulas. The pattern for each cell should be:

```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```

where:
- `<data_range>` is the rectangular block of numeric values on the Data sheet (rows 21–38, columns containing the yearly data). Use absolute references (e.g., `Data!$C$21:$G$38` – adjust to actual columns).
- `<series_code_column>` is the column of series codes on Data, same rows (e.g., `Data!$A$21:$A$38` – adjust to actual column).
- `<year_row>` is the row of year headers on Data (e.g., `Data!$C$20:$G$20` – adjust to actual row and columns).
- `$D12` uses column-absolute so it locks to column D when filling across.
- `H$10` uses row-absolute so it locks to row 10 when filling down.

Write these formulas into every cell in H12:L17, H19:L24, and H26:L31. Use a loop: for each block, for each row, for each column H–L, write the formula string. Make sure to use the correct row and column references.

After writing, re-read a few cells (e.g., H12, L17, H19, L31) to confirm the formula strings are stored correctly.

## Phase 2 – Net reliability gap (H35:L40)

The formula for each cell is:
```
=(H12 - H19) / H26 * 100
```
adjusted for the correct row offsets:
- Row 35 uses data from rows 12, 19, 26 (first region)
- Row 36 uses data from rows 13, 20, 27
- Row 37 uses data from rows 14, 21, 28
- Row 38 uses data from rows 15, 22, 29
- Row 39 uses data from rows 16, 23, 30
- Row 40 uses data from rows 17, 24, 31

For each cell in H35:L40, write the formula `=(<success_cell> - <failed_cell>) / <capacity_cell> * 100` using the appropriate cell references.

## Phase 3 – Summary statistics (H42:L47)

For each column H through L:
- Row 42 (minimum): `=MIN(H35:H40)`
- Row 43 (maximum): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (simple mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

Check the labels in column D or G for rows 42–47 to confirm the ordering matches (min, max, median, mean, 25th, 75th). If the ordering differs, adjust accordingly.

## Phase 4 – Weighted mean (H50:L50)

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net reliability gap values using Compute Capacity as weights.

## Phase 5 – Save and verify

1. Save the workbook to `/root/output/result.xlsx`. Create the `/root/output/` directory if it doesn't exist.
2. Reopen the saved file (without `data_only=True`) and spot-check:
   - H12 should contain a formula string (not None).
   - H35 should contain a formula string.
   - H42 should contain a formula string.
   - H50 should contain a formula string.
3. Print these formula strings to confirm correctness.

IMPORTANT NOTES:
- Do NOT use `data_only=True` at any point.
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify any existing formatting.
- All formula strings in openpyxl must start with `=`.
- Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`).
- Double-check every Data sheet reference against your Phase 0 inspection output.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=easy, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.