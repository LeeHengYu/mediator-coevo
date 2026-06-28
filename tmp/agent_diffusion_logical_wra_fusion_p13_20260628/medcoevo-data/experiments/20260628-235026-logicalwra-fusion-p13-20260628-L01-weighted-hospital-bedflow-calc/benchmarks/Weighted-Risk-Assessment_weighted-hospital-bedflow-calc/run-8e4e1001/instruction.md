# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 – Preparation
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1 – Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and print:
- Sheet names.
- `Task` sheet: contents of column D rows 12-17, 19-24, 26-31 (series codes), row 10 columns H-L (years), and any existing content in H12:L31, H35:L47, H50:L50.
- `Data` sheet: row 20 (header) and rows 21-38, columns A-L. Identify which column holds the series codes and which columns hold year data.

This tells you the exact series-code column on Data and the year-header row on Data.

## 2 – Write lookup formulas in H12:L17, H19:L24, H26:L31
For every cell in those three blocks, write an INDEX/MATCH formula that:
- Uses the series code from column D of the **same row** on Task.
- Uses the year from row 10 of the **same column** on Task.
- Looks up against the Data sheet rows 21:38.

The exact pattern (adjust column letters if inspection shows differently):
```
=INDEX(Data!$H$21:$L$38, MATCH($D12, Data!$D$21:$D$38, 0), MATCH(H$10, Data!$H$20:$L$20, 0))
```
Adjust `$H$20:$L$20` to whichever row on Data contains the year headers (likely row 20 or row 1 – confirm from step 1). Lock references appropriately so the formula copies across the 5 columns and down the 6 rows of each block. Use absolute row/column references for the data range and year header, relative for the current row's D-column code and current column's year.

## 3 – Write Net Patient Flow formulas in H35:L40
These 6 rows correspond to 6 hospitals. The formula for each cell is:
```
=(H12 - H19) / H26 * 100
```
where H12 is Patient Admissions (block 1 same row offset), H19 is Patient Discharges (block 2 same row offset), H26 is Effective Bed Capacity (block 3 same row offset). Specifically:
- H35 = (H12-H19)/H26*100
- H36 = (H13-H20)/H27*100  … through H40 = (H17-H24)/H31*100
And similarly for columns I-L.

## 4 – Write summary statistics in H42:L47
For each column H through L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

Check the Task sheet labels in column D/E/F/G for rows 42-47 to confirm the correct order (MIN, MAX, MEDIAN, AVERAGE, 25th, 75th). Adjust row assignments if the labels differ.

## 5 – Write weighted mean in H50:L50
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## 6 – Cache numeric values into the XML
The verifier likely reads with `data_only=True` or extracts CSV, so formulas alone may show as None. After writing all formulas:

1. Save the workbook with openpyxl (this writes formulas).
2. Re-open the saved file with openpyxl (data_only=False).
3. For every cell you wrote a formula into, **evaluate the formula in Python** using the data from the Data sheet and write the numeric result as the cell's cached value. Do this by:
   a. Loading the Data sheet values into a Python dict keyed by (series_code, year).
   b. Computing each lookup result, net-flow, statistic, and weighted mean in Python.
   c. Setting `cell.value` to the computed number (float) — but **also** keeping the formula. With openpyxl you cannot have both simultaneously in `cell.value`. Instead, manipulate the XML directly:
      - After `wb.save(...)`, open the xlsx as a zip, parse `xl/worksheets/sheet1.xml` (the Task sheet), and for each formula cell add a `<v>` element with the numeric value inside the `<c>` element. Make sure the cell type attribute `t` is not set to `s` (string). Save the modified zip as the final output.
   
   Alternatively, a simpler approach: write the workbook twice:
   - First pass: write only numeric values (no formulas) and save.
   - Second pass: re-open, overwrite cells with formulas, save. openpyxl preserves the cached `<v>` from the first pass when you set a formula on a cell? **No, it does not.**
   
   **Best approach**: Use direct XML manipulation after the formula save.

### XML value injection procedure
```python
import zipfile, shutil, copy
from lxml import etree

# 1. Save formula workbook to /root/output/result_formulas.xlsx
# 2. Build dict: cell_ref -> numeric_value for all edited cells
# 3. Open the xlsx zip, find the Task sheet XML
# 4. Parse XML, find each <c r="H12"> etc., ensure <f> exists, add/replace <v> with str(numeric_value)
# 5. Remove t="s" if present on those cells
# 6. Write modified zip to /root/output/result.xlsx
```

Use the spreadsheet namespace `http://schemas.openxmlformats.org/spreadsheetml/2006/main`.

To find which sheet XML corresponds to Task, check `xl/workbook.xml` for sheet names and `xl/_rels/workbook.xml.rels` for the mapping.

## 7 – Validate
- Open `/root/output/result.xlsx` with openpyxl data_only=True and print H12:L17, H35:L40, H42:L47, H50:L50. Confirm numeric values appear (not None).
- Open with data_only=False and confirm formulas are present.
- Confirm no extra sheets were added.
- Confirm formatting is preserved (spot-check a few cells' fill colors).

## Key cautions
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Adjust all cell references based on what you actually observe in step 1. The row/column references above are best guesses; the inspection step is critical.
- If the Data sheet year headers are in a different row than 20, adjust the MATCH range accordingly.
- If the series codes on Data are in a column other than D, adjust accordingly.
- The order of statistics rows 42-47 must match the labels already in the Task sheet.

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