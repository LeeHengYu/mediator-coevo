# Task Instruction

## Task: Update workbook with formulas for Weighted Risk Assessment

### Setup
1. First, copy the source workbook: `cp /root/data/workbook.xlsx /root/output/result.xlsx`
2. Install openpyxl if needed: `pip install openpyxl`
3. Inspect the workbook structure thoroughly before making any changes.

### Inspection Phase
Open `/root/output/result.xlsx` with openpyxl and inspect:
- **Sheet `Task`**: Print the contents of rows 1-55, columns A-L. Pay special attention to:
  - Row 10 (years row) — identify what years are in H10:L10
  - Column D rows 12-17, 19-24, 26-31 (series codes)
  - Column D rows 35-40 (service names or references for Net SLA buffer)
  - What labels are in column A or nearby for rows 12-17 (block 1), 19-24 (block 2), 26-31 (block 3)
  - Rows 42-47 labels (min, max, median, mean, 25th percentile, 75th percentile)
  - Row 50 label
  - Check the fill color of cells in H12:L17 to confirm they are the yellow target cells
- **Sheet `Data`**: Print rows 1-40 thoroughly. Pay special attention to:
  - Row 21-38: the source data records
  - Identify the structure: which row has headers, which column has the series codes, which columns/rows have years
  - Understand the data layout so you can determine the correct lookup pattern

Print all cell values AND formulas for both sheets. Print row and column indices clearly.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write a formula that:
- Uses the series code from column D of that row (e.g., `$D12` for row 12 — use `$D` to lock the column so it doesn't shift across columns)
- Uses the year from row 10 of that column (e.g., `H$10` for column H — use `$10` to lock the row)
- Looks up the value from sheet `Data` rows 21:38

Choose the lookup pattern based on the Data sheet layout:
- If Data has series codes in a column and years across columns: use `INDEX(Data!range, MATCH($Dn, Data!series_column, 0), MATCH(H$10, Data!year_row, 0))`
- If Data has a different layout, adapt accordingly.

IMPORTANT: Use absolute references for the Data range and the lookup arrays. Use mixed references ($D for column lock, $10 for row lock) so formulas can be consistent across the block.

Write these as Excel formula strings using openpyxl (set `cell.value = '=FORMULA...'`). Do NOT use openpyxl's data_only mode for writing.

### Step 2: Net SLA buffer in H35:L40 and statistics in H42:L47

For H35:L40, the formula is:
`(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`

Determine which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to:
- Latency Budget Preserved
- Latency Budget Consumed  
- Covered Request Capacity

This should be evident from the labels in the Task sheet. Write cell formulas referencing the appropriate cells. For example, if block 1 is Latency Budget Preserved (rows 12-17), block 2 is Latency Budget Consumed (rows 19-24), and block 3 is Covered Request Capacity (rows 26-31), then H35 = `=(H12-H19)/H26*100`.

For H42:L47, calculate column-wise statistics over H35:L40:
- Row 42: `=MIN(H35:H40)` (or whichever row is labeled minimum)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

Match the row labels exactly — read them from the sheet to assign the correct function to the correct row.

### Step 3: Weighted mean in H50:L50

For each column H-L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of Net SLA buffer percentages weighted by Covered Request Capacity.

### Important constraints
- Do NOT modify any existing formatting (fonts, fills, borders, number formats). Use openpyxl without disturbing styles.
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- When opening the workbook, do NOT use `data_only=True`. Open normally so formulas are preserved.
- Save to `/root/output/result.xlsx`.

### Validation
After saving, reopen the file and:
1. Print cells H12:L17, H19:L24, H26:L31 to verify they contain formula strings (starting with '=')
2. Print cells H35:L40 to verify formula strings
3. Print cells H42:L47 to verify formula strings
4. Print cells H50:L50 to verify formula strings
5. Confirm no new sheets were added
6. Confirm the formulas reference the correct Data sheet ranges and use the correct lookup pattern

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