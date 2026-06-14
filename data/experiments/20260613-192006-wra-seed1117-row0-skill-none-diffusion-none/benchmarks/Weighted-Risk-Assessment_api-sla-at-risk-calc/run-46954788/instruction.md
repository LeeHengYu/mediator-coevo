# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
cp /root/data/workbook.xlsx /root/output/result.xlsx
```

## 1. Inspect the workbook structure
Open `/root/output/result.xlsx` with openpyxl and inspect:
- Sheet `Task`: Print the contents of rows 1–55, columns A–L (or at least D and H–L). Pay special attention to:
  - Row 10 (years row)
  - Column D rows 12–17, 19–24, 26–31 (series codes)
  - Rows 12–17, 19–24, 26–31 columns H–L (the yellow cells to fill with formulas)
  - Row 35–40 (Net SLA buffer rows), row 42–47 (stats rows), row 50 (weighted mean)
- Sheet `Data`: Print rows 21–38 to understand the data layout. Identify:
  - Which row/column holds series codes
  - Which row/column holds years
  - The exact range structure (is data organized with series codes in a column and years in a row, or vice versa?)

Print cell values, merged cell info, and any existing formulas. This inspection is critical before writing any formulas.

## 2. Populate H12:L17, H19:L24, H26:L31 with lookup formulas

Based on the inspection, write formulas into each cell in the three blocks. Each formula must:
- Reference the series code from column D of the same row (e.g., `$D12` for row 12)
- Reference the year from row 10 of the same column (e.g., `H$10` for column H)
- Look up the value from sheet `Data` rows 21:38
- Use one of the allowed patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH

Choose the lookup pattern based on the Data sheet layout:
- If Data has series codes in a column and years across a row header, use INDEX(MATCH, MATCH) — e.g., `=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))`
- Adapt the exact ranges based on what you found in the inspection.

IMPORTANT: Use `$D12` (column-absolute, row-relative) for the series code and `H$10` (column-relative, row-absolute) for the year, so formulas can be conceptually extended across the grid. Write each cell's formula individually since openpyxl doesn't support fill/drag.

Make sure to use the correct openpyxl method for writing formulas: assign a string starting with `=` to the cell's value.

## 3. Populate H35:L40 with Net SLA buffer formulas

The formula is: `(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`

From the inspection, identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to:
- Latency Budget Preserved
- Latency Budget Consumed  
- Covered Request Capacity

Look at labels in columns A–G to determine this. Then for each cell in H35:L40, write the appropriate formula. For example, if rows 12–17 are Latency Budget Preserved, rows 19–24 are Latency Budget Consumed, and rows 26–31 are Covered Request Capacity, then:
- H35 = `=(H12-H19)/H26*100`
- H36 = `=(H13-H20)/H27*100`
- etc.

Match the six services by their row offset within each block.

## 4. Populate H42:L47 with statistics formulas

For each column H through L:
- H42 (MIN): `=MIN(H35:H40)`
- H43 (MAX): `=MAX(H35:H40)`
- H44 (MEDIAN): `=MEDIAN(H35:H40)`
- H45 (AVERAGE/mean): `=AVERAGE(H35:H40)`
- H46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- H47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Check the labels in column A–G for rows 42–47 to confirm the exact order (MIN, MAX, MEDIAN, MEAN, 25th, 75th or some other order). Match the formula to the label.

## 5. Populate H50:L50 with weighted mean using SUMPRODUCT

For each column H through L:
- `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of Net SLA buffer percentages weighted by Covered Request Capacity.

## 6. Save and verify

Save with openpyxl. Then reopen the file and verify:
- All formula cells in the target ranges contain formula strings (start with `=`)
- No existing formatting, sheets, or other content was altered
- The file is saved at `/root/output/result.xlsx`

## Critical notes
- Use `openpyxl` with `load_workbook(filename, data_only=False)` to preserve formulas.
- Do NOT use `data_only=True` as that strips formulas.
- Write formulas as strings (e.g., `cell.value = '=INDEX(...)'`).
- Do not delete or add any sheets.
- Do not modify any cells outside the specified target ranges.
- Inspect before writing — the exact row/column layout of the Data sheet determines every formula.

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