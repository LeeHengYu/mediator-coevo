# Task Instruction

## Task: Update workbook with formulas for Weighted Risk Assessment

### Overview
You need to read `/root/data/workbook.xlsx`, populate specific cells with spreadsheet formulas on the `Task` sheet, and save to `/root/output/result.xlsx`. Do NOT add sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

### Step 0: Inspect the workbook structure
1. `mkdir -p /root/output`
2. Use openpyxl to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Read the layout carefully. Print cells in columns A-L for rows 1-55 to understand headers, labels, series codes, years, and which cells are empty (yellow targets).
   - Specifically note:
     - Row 10: which columns H-L contain year values (e.g., 2019, 2020, 2021, 2022, 2023)
     - Column D rows 12-17, 19-24, 26-31: series codes used for lookups
     - What the three blocks represent (likely three different metrics)
     - Rows 35-40: which rows correspond to which services
     - Rows 42-47: labels for min, max, median, mean, 25th, 75th percentile
     - Row 50: Platform SLA Coalition weighted mean
   - Sheet `Data`: Print rows 1-40 to understand the data layout, especially rows 21-38. Note the structure - where series codes are, where year headers are, and the data orientation (is it a vertical table with series codes in a column and years across, or vice versa?).
3. Print the exact cell references for series codes, year headers, and data range on `Data` sheet to understand lookup mechanics.

### Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas

For each cell in these ranges, create a formula that:
- Takes the series code from column D of that row on `Task` sheet
- Takes the year from row 10 of that column on `Task` sheet  
- Looks up the corresponding value from `Data` sheet rows 21:38

Use one of: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.

IMPORTANT: After inspecting the Data sheet layout, choose the appropriate lookup pattern:
- If data is arranged with series codes in a column and years across rows, INDEX(MATCH for row, MATCH for column) is most natural.
- Construct the formula referencing the `Data` sheet properly (e.g., `Data!` prefix).
- Use appropriate absolute/relative references so each cell correctly picks up its own row's series code and its own column's year.

When writing formulas with openpyxl, assign the formula as a string starting with `=`. For example: `ws['H12'] = '=INDEX(Data!$B$21:$F$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$F$20,0))'` — but adjust cell references based on actual inspection of the data layout.

### Step 2: Net SLA Buffer in H35:L40 and statistics in H42:L47

For H35:L40, the formula is:
`(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`

Determine which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to which metric by reading the labels on the Task sheet. Then construct cell references accordingly. For example, if block 1 is Latency Budget Preserved, block 2 is Latency Budget Consumed, and block 3 is Covered Request Capacity, then for H35: `=(H12-H19)/H26*100` (adjust based on actual layout).

For H42:L47, calculate column-wise statistics over H35:L40 (6 values per column):
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- MEAN: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Match each row (42-47) to the correct statistic by reading the labels in the Task sheet.

### Step 3: Weighted mean in H50:L50

For each column H-L in row 50, use SUMPRODUCT to calculate weighted mean:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This uses Net SLA Buffer percentages as values and Covered Request Capacity as weights.

### Step 4: Save and validate
1. Save to `/root/output/result.xlsx` preserving formatting. When opening with openpyxl, do NOT use `data_only=True` (you want to preserve formulas). Keep the workbook as-is except for the formula cells.
2. Re-open the saved file and verify:
   - Cells H12:L17, H19:L24, H26:L31 all contain formula strings (start with `=`)
   - Cells H35:L40 contain formula strings
   - Cells H42:L47 contain formula strings
   - Cells H50:L50 contain formula strings
   - No new sheets were added
   - Print a sample of formulas to confirm correctness

### Critical Notes
- Read the actual workbook structure BEFORE writing any formulas. The exact row/column references for the Data sheet lookup range are essential.
- When using openpyxl, formulas are written as strings. Make sure to use the correct Excel function names.
- Preserve all existing content and formatting. Only write to the specified empty/yellow cells.
- Use `$` for absolute references where needed in lookup formulas (anchor the data range and lookup arrays, but keep the series code row and year column relative).
- PERCENTILE.INC is the modern Excel equivalent of PERCENTILE. Either should work, but prefer PERCENTILE.INC if unsure.
- For the SUMPRODUCT weighted mean, the formula divides by SUM of weights, not SUMPRODUCT.

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