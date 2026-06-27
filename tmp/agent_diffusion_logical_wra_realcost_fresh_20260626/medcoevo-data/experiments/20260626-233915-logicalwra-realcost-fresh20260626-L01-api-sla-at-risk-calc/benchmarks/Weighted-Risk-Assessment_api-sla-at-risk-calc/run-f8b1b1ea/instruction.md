# Task Instruction

## Task: Populate formulas and cached values in /root/data/workbook.xlsx, save to /root/output/result.xlsx

### Preliminary Inspection
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: read rows 10-50, columns D-L to understand layout (series codes in column D, years in row 10, yellow cell ranges).
   - Sheet `Data`: read rows 21-38, columns D-L to understand the lookup source table structure (what column has series codes, what row has years, where values live).
3. Print the exact cell values of `Task!D12:D17`, `Task!D19:D24`, `Task!D26:D31` (series codes for the three blocks), and `Task!H10:L10` (year headers). Also print `Data!D21:D38` and `Data!H20:L20` or row 20-21 area to see how the Data table is structured (headers vs data).

### Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31
4. Using openpyxl (NOT data_only mode), open the workbook.
5. For each cell in the three blocks (H12:L17, H19:L24, H26:L31), write an INDEX/MATCH formula:
   ```
   =INDEX(Data!$H$21:$L$38, MATCH($D{row}, Data!$D$21:$D$38, 0), MATCH(H$10, Data!$H$20:$L$20, 0))
   ```
   **IMPORTANT**: Before writing formulas, verify the exact row that contains year headers on the Data sheet. It might be row 20 or another row. Adjust the MATCH range for years accordingly. Also verify that Data column D rows 21:38 contain the series codes. Adjust references if needed based on inspection.

   Use absolute references for the data range and mixed references ($D for the series code column, H$10 etc. for the year row) so formulas copy correctly across the grid.

### Step 2: Net SLA Buffer in H35:L40 and Summary Stats in H42:L47
6. Identify which rows in the three lookup blocks correspond to:
   - `Latency Budget Preserved` (likely H12:L17 block)
   - `Latency Budget Consumed` (likely H19:L24 block)  
   - `Covered Request Capacity` (likely H26:L31 block)
   Verify by reading labels near rows 11, 18, 25 on the Task sheet.

7. For H35:L40 (6 rows × 5 columns), write the Net SLA buffer formula. If block 1 is rows 12-17, block 2 is rows 19-24, block 3 is rows 26-31, then for cell H35:
   ```
   =(H12-H19)/H26*100
   ```
   Adjust row offsets so row 35→12,19,26; row 36→13,20,27; etc.

8. For H42:L47 (summary statistics over H35:L40), write:
   - H42: `=MIN(H35:H40)` (minimum)
   - H43: `=MAX(H35:H40)` (maximum)
   - H44: `=MEDIAN(H35:H40)` (median)
   - H45: `=AVERAGE(H35:H40)` (simple mean)
   - H46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
   - H47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)
   Each formula spans the column's 6-row range (e.g., for column I: I35:I40). Verify the labels in column D or G near rows 42-47 to confirm the order (MIN/MAX/MEDIAN/AVERAGE/P25/P75) matches what's expected.

### Step 3: Weighted Mean in H50:L50
9. For each column H through L in row 50:
   ```
   =SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
   ```
   This computes the weighted mean of Net SLA buffer using Covered Request Capacity as weights.

### Step 4: Cache Numeric Values in XML
10. Save the workbook with formulas first.
11. **Critical for verifier**: The verifier likely reads the workbook with `data_only=True` or converts to CSV, so cached `<v>` values must exist. To achieve this:
    - Use a Python approach: evaluate each formula numerically using the actual data values from the Data sheet, then write both the formula AND the cached value.
    - Specifically: read Data sheet values into a Python dict keyed by (series_code, year). Then for each formula cell, compute the numeric result in Python and set `cell.value` to the formula string, then manually set the cached value.
    - The approach: after writing all formulas, save the file. Then reopen the .xlsx as a zip, parse `xl/worksheets/sheet1.xml` (or whichever sheet is Task), and for every cell that has a formula, insert a `<v>` element with the computed numeric value. Repack the zip.
    - Alternative simpler approach: use openpyxl and after setting `cell.value = '=FORMULA'`, also do `cell._value` manipulation or use the internal `cell.data_type` approach. Actually the cleanest way: compute all values in Python, write them as `cell.value = numeric_value` first, save as a temp file, then reopen and overwrite with formulas but preserve the cached values. OR: write formulas, then post-process the XML.
    - **Recommended**: Write all formulas. Then use `zipfile` to open the saved xlsx, find the Task sheet XML, parse it, and for each `<c>` element that has an `<f>` child, add a `<v>` child with the pre-computed numeric value. Save back.

12. To compute numeric values:
    - Load Data sheet values (rows 21-38, columns D and H-L) into memory.
    - For lookup cells: find the value by matching series code and year.
    - For Net SLA buffer cells: (preserved - consumed) / capacity * 100.
    - For stats cells: compute MIN, MAX, MEDIAN, AVERAGE, PERCENTILE over the 6 buffer values per column.
    - For weighted mean: SUMPRODUCT / SUM.

### Step 5: Final Validation
13. After saving `/root/output/result.xlsx`, verify:
    - Open with openpyxl (data_only=True) and check that cells H12, H35, H42, H50 have numeric values (not None).
    - Open with openpyxl (data_only=False) and check that the same cells have formula strings starting with '='.
    - Confirm no extra sheets were added.
    - Confirm the file is valid xlsx (no corruption).
    - Print a sample of computed values for sanity checking.

### Key Warnings
- Do NOT add any new sheets, macros, VBA, or external links.
- Do NOT alter existing formatting.
- Verify the exact Data sheet layout before writing formulas—row/column references must match the actual structure.
- The `<v>` caching step is essential for the verifier to read numeric values. Do not skip it.
- If the Data sheet year headers are in a different row than 20, adjust all MATCH references accordingly.

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