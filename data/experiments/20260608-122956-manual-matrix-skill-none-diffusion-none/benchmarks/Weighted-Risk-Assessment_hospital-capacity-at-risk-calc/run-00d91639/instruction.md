# Task Instruction

Execute the following steps in order, using Python with openpyxl. Do NOT use `data_only=True` when loading or saving.

## Phase 0 — Inspect the workbook

1. Load `/root/data/workbook.xlsx` with openpyxl.
2. Print the sheet names.
3. On sheet `Data`:
   - Print rows 19–40, all columns A–Z (or however wide the data goes). Focus on:
     a. Which row contains year headers? (likely row 21 or nearby)
     b. Which column contains series codes?
     c. What are the exact series code strings in rows 21–38?
     d. What are the exact year values in the year-header row?
   - Print the cell values so you can see the full layout.
4. On sheet `Task`:
   - Print rows 1–55, columns A–L. Focus on:
     a. Row 10 — what years appear in H10:L10?
     b. Column D — what series codes appear in D12:D17, D19:D24, D26:D31?
     c. What labels are in rows 35–40 (column D or wherever the cluster names are)?
     d. What labels are in rows 42–47 (min, max, median, mean, 25th, 75th)?
     e. What is in row 50 (weighted mean row)?
   - Print everything so the layout is unambiguous.

Do NOT proceed to formula writing until you have printed and understood the full layout.

## Phase 1 — Construct lookup formulas for H12:L31

Based on your inspection, write INDEX/MATCH formulas into cells H12:L17, H19:L24, and H26:L31.

The pattern for each cell should be:
```
=INDEX(Data!<data_value_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Critical details:
- `$D12` — column-absolute reference to the series code in column D of the current row on Task sheet. Adjust the row number for each row (D12, D13, ... D17, D19, ... D24, D26, ... D31).
- `H$10` — row-absolute reference to the year in row 10. Adjust the column letter for each column (H, I, J, K, L).
- The `Data!<data_value_range>` must cover the rectangular block of numeric values in Data rows 21–38.
- The `Data!<series_code_column>` must be the single column of series codes in that same row range on Data.
- The `Data!<year_header_row>` must be the single row of year headers spanning the same columns as the data values.

Double-check that your ranges are correct by cross-referencing the printed layout. The series code column on Data and the data value columns must align (same row range). The year header row on Data and the data value rows must align (same column range).

Write each formula as a string into the cell using `ws['H12'] = '=INDEX(...)'` syntax.

## Phase 2 — Net capacity headroom (H35:L40)

The formula for each cell in H35:L40 is:
```
=(H12 - H19) / H26 * 100
```
where row 12 corresponds to "Available Care Slots" (first block), row 19 to "Occupied Care Slots" (second block), and row 26 to "Staffed Bed Capacity" (third block). Adjust row numbers for each of the 6 hospital clusters (rows 35–40 map to the 6 rows in each block: 12–17, 19–24, 26–31).

So:
- H35 = (H12 - H19) / H26 * 100
- H36 = (H13 - H20) / H27 * 100
- ...
- H40 = (H17 - H24) / H31 * 100

Adjust column letters for I, J, K, L similarly.

## Phase 3 — Statistics (H42:L47)

For each column (H through L), write:
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

IMPORTANT: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`). The previous failure had `#NAME?` errors — verify that the function name is exactly `PERCENTILE`. Also verify that the ranges H35:H40 are correct (not H35:H41 or similar).

If the labels in column A/B/C/D of rows 42–47 say something different (e.g., different order), adjust accordingly based on what you see in the inspection.

## Phase 4 — Weighted mean (H50:L50)

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net capacity headroom percentages, weighted by Staffed Bed Capacity.

## Phase 5 — Save

1. Save the workbook to `/root/output/result.xlsx`. Create the `/root/output/` directory if it doesn't exist.
2. Do NOT use `data_only=True` at any point.
3. Do NOT add any new sheets, macros, VBA, external links, or helper tabs.

## Phase 6 — Verify

1. Reload the saved file (without `data_only=True`).
2. Print the formulas in cells H12, L17, H19, L24, H26, L31, H35, L40, H42, H46, H47, H50, L50.
3. Confirm they are formula strings (start with `=`), not None or literal values.
4. Print the Task sheet's row 10 (H10:L10) to confirm years are present.
5. Print D12:D17, D19:D24, D26:D31 to confirm series codes are present.

If any cell is None or doesn't contain a formula, investigate and fix before finishing.

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