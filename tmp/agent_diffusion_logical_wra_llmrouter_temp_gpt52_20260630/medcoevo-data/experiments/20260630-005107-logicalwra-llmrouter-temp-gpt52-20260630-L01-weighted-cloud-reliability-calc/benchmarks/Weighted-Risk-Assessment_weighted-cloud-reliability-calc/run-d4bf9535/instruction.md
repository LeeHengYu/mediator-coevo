# Task Instruction

## Task: Populate formulas and cached values in /root/data/workbook.xlsx, save to /root/output/result.xlsx

### Overview
You must open `/root/data/workbook.xlsx`, add spreadsheet formulas to specific cells on the `Task` sheet, and save the result to `/root/output/result.xlsx`. The workbook has two sheets: `Task` and `Data`. Do NOT add any other sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

### Step 0: Inspect the workbook structure
1. Read the `Task` sheet to understand:
   - Column D rows 12–17, 19–24, 26–31: these contain series codes (identifiers) for lookup.
   - Row 10, columns H–L: these contain year values for lookup.
   - The yellow cells H12:L17, H19:L24, H26:L31 need lookup formulas.
   - H35:L40 need net reliability gap formulas.
   - H42:L47 need summary statistics.
   - H50:L50 needs weighted mean.
2. Read the `Data` sheet rows 21–38 to understand:
   - The layout of the source data: which column holds series codes, which row holds years, and where the numeric data lives.
   - Specifically check: What is in Data!D21:D38 (series codes)? What is in Data!H20:L20 or Data!H21:L21 area (years in a header row)? Where is the numeric data grid?

**Print out the actual cell values** for Task!D12:D31, Task!H10:L10, Data!D20:D38, Data!H20:L38 so you can construct correct formulas.

### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these ranges, write an INDEX/MATCH formula that:
- Looks up the series code from column D of the same row on `Task` sheet
- Looks up the year from row 10 of the same column on `Task` sheet
- Searches in Data sheet rows 21:38

The formula pattern for cell H12 should be something like:
```
=INDEX(Data!$H$21:$L$38, MATCH($D12, Data!$D$21:$D$38, 0), MATCH(H$10, Data!$H$20:$L$20, 0))
```
Adjust the exact row for the year header based on what you find in Step 0. The year headers might be in row 20 or another row on the Data sheet — verify this.

Apply the same formula pattern across all 90 cells (6 rows × 5 columns × 3 blocks).

### Step 2: Net reliability gap in H35:L40
For each of the 6 regions (rows 35–40) and 5 year columns (H–L):
```
= (H12 - H19) / H26 * 100
```
where the row offsets correspond to:
- Row 35 uses rows 12, 19, 26
- Row 36 uses rows 13, 20, 27
- Row 37 uses rows 14, 21, 28
- Row 38 uses rows 15, 22, 29
- Row 39 uses rows 16, 23, 30
- Row 40 uses rows 17, 24, 31

So for cell H35: `=(H12-H19)/H26*100`
For cell H36: `=(H13-H20)/H27*100`
...and so on.

### Step 3: Summary statistics in H42:L47
For each column H through L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

### Step 4: Weighted mean in H50:L50
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net reliability gap percentages using Compute Capacity as weights.

### Step 5: Cache numeric values
**Critical for verifier compatibility**: After writing all formulas, you must also cache the computed numeric values. The verifier likely reads the workbook with `data_only=True` (openpyxl) or converts to CSV, so it needs actual `<v>` (cached value) elements in the XML.

Approach:
1. First, compute all the lookup values by reading the Data sheet yourself (in Python).
2. For each formula cell, calculate what the numeric result should be.
3. After writing the formula with openpyxl, manually set the cached value by manipulating the internal XML or by using a dual-write approach:
   - Save with formulas first.
   - Then open the saved xlsx, parse the sheet XML, and inject `<v>` elements with the correct numeric values for every formula cell.
   - Re-save.

Alternatively, a cleaner approach:
1. Use openpyxl to write formulas.
2. Then use openpyxl's internal cell object: after setting `cell.value = '=FORMULA'`, also set `cell._value` to the formula but inject the cached value into the cell's XML representation by modifying the worksheet XML after save.

The most reliable method:
1. Write all formulas with openpyxl and save.
2. Reopen the .xlsx as a zip, parse `xl/worksheets/sheet1.xml` (the Task sheet), find each `<c>` element for your formula cells, and add a `<v>numeric_value</v>` child element.
3. Re-save the zip.

Make sure you compute the correct numeric values by doing the lookups and calculations in Python using the actual Data sheet values.

### Step 6: Validate
1. Reopen `/root/output/result.xlsx` with openpyxl in data_only=True mode.
2. Check that cells like H12, H35, H42, H50 return numeric values (not None).
3. Spot-check a few values against manual calculation.

### Important Notes
- Do NOT change any existing formatting, labels, or structure.
- Do NOT add sheets, macros, VBA, or external links.
- Make sure `/root/output/` directory exists before saving.
- The formula references must use the actual layout you discover in Step 0. Do not assume — verify the exact row/column positions on the Data sheet.

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