# Task Instruction

## Task: Weighted Hospital Bedflow Calculation

You need to update `/root/data/workbook.xlsx` by populating formula cells across three steps, then save to `/root/output/result.xlsx`.

### Preliminary: Inspect the workbook

1. Create `/root/output/` directory if it doesn't exist.
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Read the structure carefully. Print cells A10:L50 to understand layout, especially:
     - Column D (series codes for each row)
     - Row 10 (years in columns H through L)
     - The three blocks H12:L17, H19:L24, H26:L31 (what metric each block represents)
     - Row 35:40 labels and structure
     - Rows 42:47 labels (min, max, median, mean, 25th, 75th percentile)
     - Row 50 (weighted mean for MHN)
   - Sheet `Data`: Print rows 21:38 to understand the data layout. Identify:
     - How the data is organized (rows vs columns)
     - Where series codes appear
     - Where years appear
     - The exact structure needed for lookup formulas
   - Also print the first few rows of `Data` sheet to understand headers and column layout.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write a spreadsheet formula (not Python computation) that:
- Uses the series code from column D of that row on sheet `Task`
- Uses the year from row 10 of the corresponding column on sheet `Task`
- Looks up the value from sheet `Data` rows 21:38
- Uses one of these patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH

IMPORTANT: You must determine the exact data layout on the `Data` sheet before writing formulas. The formula references must match the actual structure. For example, if Data has series codes in a column and years in a row, INDEX(MATCH, MATCH) is likely the cleanest approach.

When writing formulas with `openpyxl`, note:
- Formulas are strings starting with `=`
- Use absolute references for the Data range (e.g., `Data!$A$21:$A$38`) where appropriate
- Use mixed references so formulas can be consistent across the block (lock the column for series code reference, lock the row for year reference)
- Cross-sheet references use the syntax `Data!A1`

### Step 2: Net patient flow and summary statistics in H35:L40 and H42:L47

For H35:L40 (Net patient flow for 6 hospitals):
- Formula: `(Patient Admissions - Patient Discharges) / Effective Bed Capacity * 100`
- Patient Admissions should be from the first block (H12:L17)
- Patient Discharges from the second block (H19:L24)
- Effective Bed Capacity from the third block (H26:L31)
- Each row corresponds to the same hospital across all three blocks.

For H42:L47 (column-wise summary statistics over H35:L40):
- Row for MIN: `=MIN(H35:H40)` etc.
- Row for MAX: `=MAX(H35:H40)` etc.
- Row for MEDIAN: `=MEDIAN(H35:H40)` etc.
- Row for MEAN (simple): `=AVERAGE(H35:H40)` etc.
- Row for 25th percentile: `=PERCENTILE(H35:H40, 0.25)` etc.
- Row for 75th percentile: `=PERCENTILE(H35:H40, 0.75)` etc.
- Match each statistic to the correct row based on the labels in column D/E/F/G.

### Step 3: Weighted mean in H50:L50

For each column H through L:
- Use SUMPRODUCT to compute weighted mean
- Values: Net patient flow percentages from H35:H40 (Step 2 results)
- Weights: Effective Bed Capacity from H26:H31 (Step 1 results)
- Formula pattern: `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

### Saving

- Do NOT change any existing formatting, sheet names, or add new sheets.
- Save the workbook to `/root/output/result.xlsx`.

### Validation

After saving, re-open `/root/output/result.xlsx` with openpyxl and verify:
1. All target cells contain formula strings (starting with `=`), not hardcoded values.
2. The formulas reference the correct ranges.
3. No new sheets were added.
4. Print a sample of the formulas for each block to confirm correctness.

### Critical Notes
- Read the actual cell contents and layout BEFORE writing any formulas. Do not assume positions.
- The labels in the Task sheet (column D or nearby columns) and Data sheet must be cross-referenced carefully.
- Use `data_only=False` when opening with openpyxl so you work with formulas.
- Preserve all existing cell formatting by not touching cells outside the target ranges.
- If the Data sheet has the series code in column A and years as column headers, adapt your INDEX/MATCH formula accordingly.

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