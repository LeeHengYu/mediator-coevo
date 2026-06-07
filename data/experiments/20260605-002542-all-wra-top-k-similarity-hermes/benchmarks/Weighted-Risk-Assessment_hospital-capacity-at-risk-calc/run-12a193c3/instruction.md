# Task Instruction

Execute the following steps carefully to complete the task.

## 0. Inspect the workbook and test harness

```bash
cp /root/data/workbook.xlsx /root/output/result.xlsx
pip install openpyxl
```

Open `/root/output/result.xlsx` with openpyxl (data_only=False) and inspect:
- Sheet `Task`: print rows 10-50, columns D-L, to understand the layout (series codes in column D, years in row 10, yellow target cells, existing formulas if any).
- Sheet `Data`: print rows 21-38 fully (all columns) to understand the data layout (column headers, row labels, orientation).
- Check what is in rows 42-47 labels (min, max, median, mean, 25th, 75th percentile) and row 50 label.
- Also read `/tests/test_outputs.py` (or any test file present) to understand the verifier expectations.

Print everything clearly before making any edits.

## 1. Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in those ranges, write a formula that looks up data from sheet `Data` rows 21:38. The formula must use the series code from column D of the same row and the year from row 10 of the same column.

Use `INDEX(MATCH,MATCH)` pattern since it is the most robust:
```
=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```
Adjust the exact ranges after inspecting the Data sheet layout. The key points:
- The first MATCH finds the row by matching the series code in column D against the row labels in the Data sheet.
- The second MATCH finds the column by matching the year in row 10 against the column headers in the Data sheet.
- Anchor references appropriately ($D12 for mixed reference, H$10 for mixed reference).

## 2. Net capacity headroom formulas in H35:L40

These six rows correspond to six hospital clusters. Based on the task description:
- Available Care Slots = values from H12:L17
- Occupied Care Slots = values from H19:L24  
- Staffed Bed Capacity = values from H26:L31

Formula for each cell (e.g., H35): `=(H12-H19)/H26*100`

Apply this pattern for all 6 rows × 5 columns (H35:L40).

## 3. Statistics in H42:L47

For each column (H through L), calculate column-wise statistics over the 6 headroom values (rows 35:40):
- Row 42 (Minimum): `=MIN(H35:H40)`
- Row 43 (Maximum): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` — **use PERCENTILE, not PERCENTILE.INC or PERCENTILE.EXC** (cross-task artifacts warn that dotted function names cause #NAME? errors in openpyxl/verifier)
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Do NOT use `PERCENTILE.INC`, `PERCENTILE.EXC`, `QUARTILE.INC`, or any dotted function names. Use the legacy `PERCENTILE` function to avoid #NAME? errors. Similarly use `MEDIAN` not `MEDIAN.INC`. Check the verifier expectations from the cross-task failure signatures.

Also verify the row order (min/max/median/mean/25th/75th) by checking the labels in column D or the test file. Adjust if labels differ.

## 4. Weighted mean in H50:L50

For each column: `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This computes the weighted mean of the headroom percentages using Staffed Bed Capacity as weights.

## 5. Save and verify

Save the workbook to `/root/output/result.xlsx` using openpyxl. Do NOT change any formatting, do NOT add sheets.

After saving, re-open the file and print all formula cells to confirm they are correctly written. Then run the test suite if available:
```bash
cd /root && python -m pytest tests/ -v 2>&1 | head -80
```

If any #NAME? errors appear, replace dotted function names with legacy equivalents. If any values are wrong, re-inspect the Data sheet layout and adjust ranges.

## Important notes
- Before writing any formulas, thoroughly inspect both sheets to get exact column/row ranges.
- The Data sheet row/column orientation matters — determine whether years are in rows or columns.
- Preserve all existing formatting (fonts, colors, borders, etc.).
- Use openpyxl to write formulas as strings (e.g., cell.value = '=INDEX(...)').
- Do not use data_only=True when writing; open normally.
- After each major step, verify by printing what was written.

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