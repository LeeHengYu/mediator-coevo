# Task Instruction

Complete the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Phase 0 – Inspect the workbook

1. `mkdir -p /root/output`
2. Use Python with openpyxl to open `/root/data/workbook.xlsx` and print:
   a. All sheet names.
   b. From sheet `Task`: print every cell value in columns A–L for rows 1–55 (value and coordinate). Pay special attention to:
      - Row 10 (years in H10:L10)
      - Column D rows 12–31 (series codes)
      - Column A or B rows 12–31 (block/section labels)
      - Rows 35–50 labels in columns A–G
      - Any labels in rows 42–47 (min, max, median, mean, percentiles)
      - The yellow-highlighted cells if fill color is detectable
   c. From sheet `Data`: print all cell values in rows 1–40, all used columns, to understand the lookup table structure (especially rows 21:38). Identify how series codes map to rows and how years map to columns.

## Phase 1 – Populate lookup formulas (Step 1)

Based on the inspection, populate cells H12:L17, H19:L24, and H26:L31 on sheet `Task` with lookup formulas.

For each cell in these ranges:
- The lookup key 1 is the series code in column D of the same row (e.g., $D12 for row 12).
- The lookup key 2 is the year in row 10 of the same column (e.g., H$10 for column H).
- The data source is sheet `Data`, rows 21:38.

Use INDEX+MATCH+MATCH pattern:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```

Determine the exact ranges from the inspection:
- `<data_range>`: the rectangular block of numeric values in Data rows 21:38 (excluding header row/column)
- `<series_code_column>`: the column in Data sheet containing the series codes (likely column A or B in rows 21:38)
- `<year_row>`: the row in Data sheet containing the year headers

Use absolute references for the data range and lookup arrays ($), and mixed references so the formula can be consistent across the block. Lock the row for the year reference (H$10) and lock the column for the series code ($D12).

Write these formulas using openpyxl by setting each cell's `.value` to the formula string. Do NOT use `data_only` mode. Make sure the formula strings use the correct Excel syntax.

## Phase 2 – Net production slack (Step 2)

In H35:L40, for each of the 6 plants (rows 35–40) and each year column (H–L), enter a formula:
```
=(H12 - H19) / H26 * 100
```
where H12 is from the Finished Output block (rows 12–17), H19 from Scrap And Rework block (rows 19–24), and H26 from Rated Production Capacity block (rows 26–31). Adjust row references to match each plant. Verify from inspection which block is which – the three blocks in rows 12-17, 19-24, 26-31 correspond to three different data series groups. Identify which is Finished Output, Scrap And Rework, and Rated Production Capacity from the labels.

In H42:L47, enter column-wise summary statistics formulas. Identify the exact statistic for each row from the labels (inspect rows 42–47). Expected formulas:
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- AVERAGE: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- 75th percentile: `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`

Match each formula to the correct row based on the label found during inspection.

## Phase 3 – Weighted mean (Step 3)

In H50:L50, enter a SUMPRODUCT formula for the weighted mean:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the Net production slack percentages (H35:H40) as values and Rated Production Capacity (H26:H31) as weights. Adjust column letter for each of H through L.

## Phase 4 – Save and validate

1. Save the workbook to `/root/output/result.xlsx`. Do NOT use `data_only=True` when loading. Preserve existing formatting – do not modify fonts, fills, borders, number formats, or any cells outside the target ranges.
2. Re-open the saved file and verify:
   a. All target cells contain formula strings (not None or bare values).
   b. The formulas reference the correct sheets and ranges.
   c. No extra sheets were added.
   d. Print a sample of formulas from each block for confirmation.

## Critical constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify any cells outside the specified target ranges.
- Do NOT change existing formatting (fills, fonts, borders, number formats).
- Use openpyxl for all operations. When writing formulas, assign the formula string directly to cell.value (e.g., `ws['H12'] = '=INDEX(...)'`).
- If openpyxl version issues arise with formula writing, ensure you are not in data_only or read_only mode.

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