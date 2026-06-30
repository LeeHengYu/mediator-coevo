# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

Follow these steps exactly:

### Step 0: Inspect the workbook
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and inspect:
   - Sheet `Task`: read row 10 (especially H10:L10) to find the year headers.
   - Read column D rows 12-17, 19-24, 26-31 to find the series codes for each block.
   - Read H35 area to see if there are already labels or structure for Net production slack.
   - Read rows 42-47 to see labels for MIN/MAX/MEDIAN/AVERAGE/PERCENTILE.
   - Read row 50 to see the label for Regional Output Council.
   - Sheet `Data`: read rows 21-38, especially column D (series codes) and columns H-L (data values). Also read row 20 or whichever row has year headers for the Data sheet.
   - Print all of these so you understand the exact layout.

### Step 1: Write lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write an INDEX/MATCH formula that:
- Looks up the series code from column D of the current row (on Task sheet) AND the year from row 10 (on Task sheet)
- Searches in Data sheet rows 21:38
- Pattern: `=INDEX(Data!$H$21:$L$38, MATCH($D{row}, Data!$D$21:$D$38, 0), MATCH(H$10, Data!$H$20:$L$20, 0))`

Adjust the Data year-header row reference based on what you actually find when inspecting. The column reference for MATCH should use the year headers row on the Data sheet. The row reference for MATCH should use Data column D (series codes) rows 21:38.

IMPORTANT: Use `$D{row}` (absolute column, relative row) for the series code, and `H$10` (relative column, absolute row) for the year, so formulas copy correctly across the range.

Write these using openpyxl by setting `cell.value = '=INDEX(...)'` as a string.

### Step 2: Net production slack in H35:L40

Based on the layout you discover, the three blocks are:
- H12:L17 = one metric (e.g., Finished Output)
- H19:L24 = another metric (e.g., Scrap And Rework)  
- H26:L31 = another metric (e.g., Rated Production Capacity)

Check column D rows 12-17, 19-24, 26-31 to identify which block corresponds to which metric. The formula is:
`Net production slack = (Finished Output - Scrap And Rework) / Rated Production Capacity * 100`

For each cell in H35:L40 (6 rows × 5 columns), write a formula referencing the corresponding cells from the three blocks. For example, if Finished Output is in rows 12-17, Scrap And Rework in rows 19-24, and Rated Production Capacity in rows 26-31, then:
`H35 = (H12 - H19) / H26 * 100`
`H36 = (H13 - H20) / H27 * 100`
etc.

Adjust row references based on actual block-to-metric mapping you discover.

### Step 3: Statistics in H42:L47

For each column H through L:
- Row 42: `=MIN(H35:H40)` (or whatever the correct column is)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

Check the labels in column D/E/F/G for rows 42-47 to confirm the correct order (MIN, MAX, MEDIAN, AVERAGE, 25th percentile, 75th percentile). Adjust the row assignments to match the actual labels.

### Step 4: Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses Net production slack values as the values and Rated Production Capacity as weights. Again, adjust the Rated Production Capacity reference based on which block it actually is.

### Step 5: Cache numeric values

The verifier likely reads values with openpyxl data_only=True or CSV export, which requires cached values. After writing all formulas:

1. Save the workbook with openpyxl first to `/root/output/result.xlsx`.
2. Then open it with a formula evaluation approach. Use one of these methods:
   a. Try using `subprocess` to run LibreOffice in headless mode to open and re-save: `libreoffice --headless --calc --convert-to xlsx --outdir /root/output /root/output/result.xlsx` (may need to rename first to avoid overwrite issues).
   b. If LibreOffice is not available, manually compute the numeric values in Python and inject them as cached values by manipulating the cell objects. For each formula cell, compute the expected numeric result and set it via the internal `_value` or by writing to the XML.

The preferred approach: After saving with openpyxl, try LibreOffice headless conversion. Check if `libreoffice` or `soffice` is available first with `which libreoffice` or `which soffice`.

If LibreOffice is available:
```
cp /root/output/result.xlsx /root/output/temp_result.xlsx
libreoffice --headless --calc --convert-to xlsx:".xlsx" --outdir /root/output /root/output/temp_result.xlsx
mv /root/output/temp_result.xlsx /root/output/result.xlsx
```

If LibreOffice is NOT available, you must manually compute cached values:
- Read the Data sheet values into a Python dict keyed by (series_code, year).
- For each formula cell, compute the numeric result in Python.
- After computing, set both the formula and the cached value. With openpyxl, you can do this by writing the formula, saving, then reopening and using the internal XML manipulation to add `<v>` elements. Alternatively, use the approach of writing formulas AND separately computing values, then patching the xlsx XML via zipfile manipulation.

Actually, the most reliable approach for caching: 
- Compute all values in Python.
- Write formulas to cells.
- Save workbook.
- Then use `openpyxl` to reopen, and for each cell that has a formula, also store the computed value by manipulating the worksheet's internal XML. The cleanest way: after `wb.save()`, use zipfile to open the xlsx, parse `xl/worksheets/sheet1.xml`, find each formula cell `<c>`, and add a `<v>numeric_value</v>` element inside it.

### Step 6: Validate
- Reopen the saved file with openpyxl (data_only=True) and print cells H12, H35, H42, H50 to verify cached values are present.
- Reopen with data_only=False and print the same cells to verify formulas are present.
- Confirm no extra sheets were added.

### Critical Notes
- Do NOT add any new sheets, macros, VBA, or external links.
- Do NOT change existing formatting.
- The final file must be at `/root/output/result.xlsx`.
- Formulas must use INDEX/MATCH (or one of the allowed lookup patterns).
- Inspect everything before writing. Do not assume row/column positions.

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