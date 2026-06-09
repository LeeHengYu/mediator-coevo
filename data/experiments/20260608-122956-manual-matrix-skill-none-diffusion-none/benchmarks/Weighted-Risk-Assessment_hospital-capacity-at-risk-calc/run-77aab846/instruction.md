# Task Instruction

## Task: Update hospital capacity workbook with formulas

### Overview
You need to read, understand, and update `/root/data/workbook.xlsx` by populating specific cells with spreadsheet formulas, then save the result to `/root/output/result.xlsx`.

### Step 0: Inspect the workbook structure
1. `mkdir -p /root/output`
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Print rows 1-55 with all values and formulas. Pay special attention to:
     - Column D (series codes) for rows 12-17, 19-24, 26-31
     - Row 10 (years in columns H through L)
     - The labels/headers for rows 35-40, 42-47, 50
     - Any existing formulas or values already present
     - Cell fill colors to confirm yellow cells
   - Sheet `Data`: Print rows 21-38 fully. Understand the layout — which row/column contains what. Note the exact structure: are series codes in a column? Are years in a row? Identify the lookup table dimensions precisely.
3. Print the exact cell references and their content for columns A-L, rows 10-50 on sheet `Task` so you have a complete picture.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a spreadsheet formula (not a Python-computed value) that:
- Takes the series code from column D of that row
- Takes the year from row 10 of that column
- Looks up the corresponding value from sheet `Data` rows 21:38
- Uses one of the allowed patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH

IMPORTANT: 
- Use `openpyxl` to write formulas as strings (e.g., `ws['H12'] = '=INDEX(...)'`)
- Make sure references to the Data sheet use the correct syntax: `Data!` prefix
- Examine the Data sheet layout carefully to determine whether series codes are in rows or columns, and whether years are in rows or columns, then choose the appropriate lookup pattern
- Use absolute references (with $) where appropriate for the lookup range so formulas can be filled across columns/rows correctly
- Verify the formula pattern works by checking that the lookup dimensions match

### Step 2: Net capacity headroom in H35:L40 and statistics in H42:L47

For H35:L40 (6 rows × 5 columns):
- Formula: `(Available Care Slots - Occupied Care Slots) / Staffed Bed Capacity * 100`
- Determine which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to Available Care Slots, Occupied Care Slots, and Staffed Bed Capacity by reading the row labels
- Write cell formulas referencing the appropriate cells from those blocks

For H42:L47 (statistics), write column-wise formulas over H35:L40:
- Check the labels in column A/B/C/D for rows 42-47 to determine which statistic goes where
- MIN: `=MIN(H35:H40)` pattern
- MAX: `=MAX(H35:H40)` pattern  
- MEDIAN: `=MEDIAN(H35:H40)` pattern
- AVERAGE (simple mean): `=AVERAGE(H35:H40)` pattern
- 25th percentile: `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`
- Match each to the correct row based on the label

### Step 3: Weighted mean in H50:L50

For each column H through L in row 50:
- Use SUMPRODUCT with the Net capacity headroom values (H35:H40 for column H, etc.) as values and the Staffed Bed Capacity values (H26:H31 for column H, etc.) as weights
- Formula pattern: `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

### Step 4: Save and validate
1. Save the workbook to `/root/output/result.xlsx` preserving all formatting
2. Reopen the saved file and verify:
   - All formula cells in the target ranges contain formula strings (start with `=`)
   - No sheets were added or removed
   - The formulas reference correct ranges
   - Print a sample of formulas from each section to confirm correctness
3. Do NOT use `data_only=True` when writing — formulas must be preserved as formulas

### Critical constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs
- Do NOT change existing formatting (fonts, fills, borders, etc.)
- Do NOT overwrite non-yellow cells
- All formulas must be Excel spreadsheet formulas written as strings, not Python-computed values
- Use openpyxl for all Excel operations

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