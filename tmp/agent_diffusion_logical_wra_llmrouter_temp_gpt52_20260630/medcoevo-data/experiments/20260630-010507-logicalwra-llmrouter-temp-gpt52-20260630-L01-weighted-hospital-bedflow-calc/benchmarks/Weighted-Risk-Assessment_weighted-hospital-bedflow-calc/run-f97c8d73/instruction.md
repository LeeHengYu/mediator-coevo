# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 — Inspect the workbook
```bash
mkdir -p /root/output
```
Open `/root/data/workbook.xlsx` with openpyxl (NOT data_only) and print:
- Sheet names.
- Task!D12:D17 (series codes for block 1), Task!D19:D24 (block 2), Task!D26:D31 (block 3).
- Task!H10:L10 (year headers).
- Data!A21:A38 and Data!D21:D38 (row labels and series codes on Data sheet).
- Data!H20:L20 or Data!H1:L1 (column headers on Data sheet — find the year row).
- Data!H21:L38 first few rows of numeric data.
- Task!D35:D40 (hospital names for net-flow block).
- Task!H26:L31 current content (will hold Effective Bed Capacity after Step 1).
- Task!D42:D47 (stat labels for min/max/median/mean/p25/p75).
- Task!D50 (label for weighted mean row).

Print everything so you can build correct formulas.

## 1 — Write lookup formulas (Step 1)

For every cell in H12:L17, H19:L24, H26:L31 on sheet `Task`, write a formula that looks up the value from `Data` sheet rows 21–38. Use `INDEX/MATCH` (preferred) or another allowed pattern.

The formula pattern for cell (r, c) should be:
```
=INDEX(Data!$H$21:$L$38, MATCH($D{r}, Data!$D$21:$D$38, 0), MATCH(H$10, Data!$H$20:$L$20, 0))
```
Adjust the year-header row reference (`$H$20:$L$20`) to wherever the year labels actually are on the Data sheet (inspect first!). The `$D{r}` is the series-code cell in column D of the current row. `H$10` (or I$10, J$10 …) is the year from row 10 of Task. Make sure mixed references are correct so the formula can span H–L columns and the appropriate rows.

Write these formulas using openpyxl by iterating over the three row-ranges and columns H(8) through L(12).

## 2 — Net patient flow formulas (Step 2, H35:L40)

For each hospital row i (0–5), the net patient flow is:
```
=(H{admissions_row} - H{discharges_row}) / H{capacity_row} * 100
```
where admissions_row, discharges_row, capacity_row correspond to the same hospital in blocks H12:L17 (Patient Admissions), H19:L24 (Patient Discharges), H26:L31 (Effective Bed Capacity). So for row 35 col H:
```
=(H12-H19)/H26*100
```
Row 36: `=(H13-H20)/H27*100`, etc. through row 40.

Write these formulas for each cell in H35:L40.

## 3 — Summary statistics (Step 2, H42:L47)

For each column c in H–L, write in rows 42–47:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

Check the actual stat labels in D42:D47 first and match the order to whatever labels are there (e.g., if D42 says "Maximum" put MAX in row 42). Adjust accordingly.

## 4 — Weighted mean (Step 3, H50:L50)

For each column c in H–L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## 5 — Inject cached numeric values

The verifier may read the workbook in data_only mode (openpyxl) or convert to CSV, which requires cached `<v>` values. After writing all formulas:

1. Also read Data sheet values (load workbook again with data_only or just read the already-loaded non-formula cells if Data has plain numbers).
2. For every formula cell you wrote, compute the expected numeric value in Python using the actual data from the Data sheet.
3. Use the internal openpyxl cell attribute `cell._value` to keep the formula, but also set `cell.value` to the formula string AND patch the cached value into the worksheet XML after saving. 

The most reliable approach:
- Save the workbook first with formulas via openpyxl.
- Then open the saved .xlsx as a zip, parse `xl/worksheets/sheet1.xml` (the Task sheet), and for every cell that has a `<f>` element, add a `<v>` element with the computed numeric value.
- Re-pack the zip to `/root/output/result.xlsx`.

To compute values: load the Data sheet values into a dict keyed by (series_code, year) → number. Then for lookup cells, just do the lookup in Python. For net-flow cells, compute from the looked-up values. For stats, compute from the net-flow values. For weighted mean, compute from net-flow and capacity values.

## 6 — Validate

- Reopen `/root/output/result.xlsx` with openpyxl (data_only=True) and print H12, H19, H26, H35, H42, H50 to confirm cached values are present.
- Reopen with data_only=False and print the same cells to confirm formulas are present.
- Confirm no extra sheets were added.
- Confirm formatting is preserved (spot-check a few cells' fill colors).

## Critical notes
- Do NOT add any new sheets, macros, VBA, or external links.
- Do NOT alter any existing formatting (fonts, fills, borders, column widths).
- Verify the exact row on the Data sheet where year headers live before writing formulas.
- If the stat label order in D42:D47 differs from min/max/median/mean/p25/p75, match the labels exactly.
- The formula references must use the correct absolute/mixed references so they work across all cells in each block.

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