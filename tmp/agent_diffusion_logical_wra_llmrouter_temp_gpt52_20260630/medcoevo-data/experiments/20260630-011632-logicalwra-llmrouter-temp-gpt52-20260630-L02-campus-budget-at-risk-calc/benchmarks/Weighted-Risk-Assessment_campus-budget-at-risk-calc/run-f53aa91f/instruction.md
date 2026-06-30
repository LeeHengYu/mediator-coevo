# Task Instruction

## Task: Populate formulas and cached values in /root/data/workbook.xlsx, save to /root/output/result.xlsx

### Overview
You must open `/root/data/workbook.xlsx`, populate specific formula cells on the `Task` sheet using data from the `Data` sheet, and save the result to `/root/output/result.xlsx`. The verifier likely reads cached cell values (not live formula evaluation), so after writing formulas you must also inject numeric cached values.

### Step 0: Inspect the workbook structure
1. `mkdir -p /root/output`
2. Open the workbook with openpyxl (NOT data_only) and inspect:
   - `Task` sheet: read row 10 (the year headers in columns H–L), read column D rows 12–17, 19–24, 26–31 (series codes for each block), read rows 35–40 column D (department names or codes).
   - `Data` sheet: read rows 21–38 entirely. Identify the structure: which column holds series codes, which row holds years, where numeric data lives (likely columns H–L or similar). Print all of this so you understand the exact layout.
   - Also inspect what's in H12:L17, H19:L24, H26:L31 currently (should be empty/yellow).
   - Inspect H35:L47 and H50:L50 current state.
   - Print the exact column letters and row numbers for all relevant ranges.

### Step 1: Write lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three 6×5 blocks, write a formula that looks up:
- The series code from column D of that row on `Task`
- The year from row 10 of that column on `Task`
- Against the data table on `Data` sheet rows 21–38

Use `INDEX/MATCH` pattern. The exact references depend on what you find in Step 0. A typical formula for cell H12 would be something like:
```
=INDEX(Data!$H$21:$L$38, MATCH($D12, Data!$D$21:$D$38, 0), MATCH(H$10, Data!$H$20:$L$20, 0))
```
Adjust the exact row/column references based on your inspection. The MATCH for the year header should reference the row on `Data` that contains year labels (inspect to find it — could be row 20 or another row). The MATCH for the series code should reference the series code column on `Data` (likely column D or similar).

Write all 90 cells (3 blocks × 6 rows × 5 columns).

### Step 2: Write Net budget buffer formulas in H35:L40
The formula is: `(Committed Funding - Operating Spend) / Approved Budget Base * 100`

Based on the three lookup blocks:
- Block 1 (H12:L17) corresponds to one metric (e.g., Committed Funding)
- Block 2 (H19:L24) corresponds to another metric (e.g., Operating Spend)  
- Block 3 (H26:L31) corresponds to Approved Budget Base

Inspect the labels in the Task sheet (likely in column A or nearby) to determine which block is which. Then for H35 the formula would be something like:
```
=(H12 - H19) / H26 * 100
```
(Adjust row references so each of the 6 department rows maps correctly.)

Write all 30 cells (6 rows × 5 columns).

### Step 2b: Write summary statistics in H42:L47
For each column (H through L), compute over the 6 values in rows 35–40:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

Check the Task sheet labels near rows 42–47 to confirm the correct order of these statistics. Adjust the order to match whatever labels are shown.

### Step 3: Write weighted mean in H50:L50
For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the Net budget buffer percentages as values and Approved Budget Base as weights.

### Step 4: Inject cached numeric values
This is CRITICAL for the verifier. After writing all formulas:

1. Read the Data sheet values to get the actual numeric source data.
2. For every formula cell you wrote, compute the expected numeric result in Python.
3. Use openpyxl to set `cell.value` to the formula string (already done), but also you need to ensure the cached value is present in the saved XML.

openpyxl does not natively support setting cached values easily. Use this approach:
- Save the workbook first with openpyxl.
- Then open the saved .xlsx as a zip, parse `xl/worksheets/sheet1.xml` (or whichever sheet is Task), and for each formula cell, add a `<v>` element with the computed numeric value inside the `<c>` element.
- Rewrite the zip with the modified XML.

Alternatively, a simpler approach: after writing formulas with openpyxl, for each cell, also set `cell._value` and manipulate the cell's internal state. But the zip/XML approach is more reliable.

Here is the recommended XML injection approach:
1. Save with openpyxl to `/root/output/result.xlsx`.
2. Open the xlsx as a zipfile.
3. Parse the Task sheet XML.
4. For each cell reference (e.g., "H12"), find the `<c>` element, ensure it has `t` attribute removed (or not set to "s"), and add/replace a `<v>` child with the numeric string.
5. Save the modified zip back to `/root/output/result.xlsx`.

### Step 5: Validate
1. Reopen `/root/output/result.xlsx` with openpyxl in data_only=True mode.
2. Read all formula cells and confirm they return numeric values (not None).
3. Spot-check a few values against manual Python calculations.
4. Confirm no extra sheets were added, no macros, no external links.

### Important Notes
- Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
- Preserve all existing formatting.
- The exact row that contains year headers on the Data sheet, and the exact column that contains series codes, must be determined by inspection. Do not assume.
- Verify the mapping of blocks to metrics (Committed Funding, Operating Spend, Approved Budget Base) by reading labels on the Task sheet.
- If the statistics rows 42–47 have labels, match the formula to the label (don't assume the order I gave above is correct).
- When injecting cached values into XML, be careful with the namespace handling in the spreadsheetML XML.

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