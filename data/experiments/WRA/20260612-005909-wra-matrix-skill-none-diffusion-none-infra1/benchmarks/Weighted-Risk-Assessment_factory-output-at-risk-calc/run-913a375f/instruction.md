# Task Instruction

Execute the following steps to produce /root/output/result.xlsx.

## 0 – Preparation
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1 – Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Print:
- Sheet names.
- On sheet `Task`: the contents of cells D12:D17, D19:D24, D26:D31 (series codes), row 10 from H10:L10 (years), the labels in G42:G47 (stat names), and the label in G50 (should mention Regional Output Council or similar). Also print the exact text in cells G35:G40 to understand the plant names/order.
- On sheet `Data`: print rows 20–38 fully (all columns A through the last used column) so you can see the header row and data layout. Also print row 1 or whatever row contains the year headers, and column A or B values to understand the structure.

Print everything clearly so you can map references precisely.

## 2 – Write the lookup formulas (Step 1)
Using the inspection results, write `INDEX(MATCH,MATCH)` formulas into the yellow cells.

For each cell in H12:L17, H19:L24, and H26:L31:
- The formula pattern is: `=INDEX(data_range, MATCH(series_code, series_column, 0), MATCH(year, year_row, 0))`
- `series_code` = the cell in column D of the current row (e.g., D12 for row 12).
- `year` = the cell in row 10 of the current column (e.g., H10 for column H).
- `data_range` = the rectangular block on sheet `Data` that contains the numeric values in rows 21:38 (adjust based on inspection — identify the exact rows and columns that hold the data values, excluding the header row and the series-code column).
- `series_column` = the column on `Data` that contains the series codes (the leftmost label column within rows 21:38).
- `year_row` = the row on `Data` that contains the year headers (the row just above the data, within the header area).

Use absolute references for the Data sheet ranges (with `$`) and mixed references for the lookup keys so formulas copy correctly.

IMPORTANT: Prefix Data sheet references with `Data!` (e.g., `Data!$A$21:$A$38`).

## 3 – Write Net production slack formulas (Step 2 top block)
For each cell in H35:L40, write:
```
=(Hxx - Hyy) / Hzz * 100
```
where:
- `Hxx` = the corresponding Finished Output cell (from the first lookup block, H12:L17),
- `Hyy` = the corresponding Scrap And Rework cell (from the second lookup block, H19:L24),
- `Hzz` = the corresponding Rated Production Capacity cell (from the third lookup block, H26:L31).

The six plants in rows 35–40 correspond to the six plants in rows 12–17 (same order). So H35 = (H12 - H19) / H26 * 100, H36 = (H13 - H20) / H27 * 100, etc.

## 4 – Write summary statistics (Step 2 bottom block)
In rows 42–47, for each column H through L:
- Row 42 (Min): `=MIN(H35:H40)`
- Row 43 (Max): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=_xlfn.PERCENTILE.INC(H35:H40,0.25)`
- Row 47 (75th percentile): `=_xlfn.PERCENTILE.INC(H35:H40,0.75)`

**CRITICAL**: For percentile formulas, you MUST use the `_xlfn.` prefix (i.e., `_xlfn.PERCENTILE.INC`). This is required because openpyxl needs this prefix for newer Excel functions to be recognized by the evaluator. The previous execution failed because the prefix was missing, resulting in `#NAME?` errors.

Verify by inspecting the labels in G42:G47 to confirm which row is which statistic. Adjust row assignments if the labels differ from the assumed order.

## 5 – Write weighted mean formula (Step 3)
For each cell in H50:L50, write:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net production slack percentages (H35:H40) weighted by Rated Production Capacity (H26:L31).

Adjust column letters for each column (H, I, J, K, L).

## 6 – Save
Save the workbook to `/root/output/result.xlsx`. Do NOT use data_only mode. Keep all existing formatting.

## 7 – Verify
Reopen the saved file and print the formulas (not values) in a few sample cells (e.g., H12, H35, H42, H46, H50) to confirm they are correctly written. Also confirm no extra sheets were added.

## Key Reminders
- Do the full structural inspection FIRST before writing any formulas.
- Use `_xlfn.PERCENTILE.INC` for percentile functions (this was the cause of the previous failure).
- Do not add sheets, macros, VBA, external links, or helper tabs.
- Do not change existing formatting.
- All formulas must use cell references, not hardcoded values.

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