# Task Instruction

You must update an Excel workbook and save the result. Follow these steps precisely.

## Setup
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx` (create `/root/output/` if needed).
2. Use `openpyxl` to open `/root/output/result.xlsx` with `data_only=False` so formulas are preserved.
3. Before making any changes, inspect the workbook thoroughly:
   - Read sheet names and confirm `Task` and `Data` exist.
   - Read the `Task` sheet: print rows 10-50 (columns A-L) to understand the layout — especially column D (series codes), row 10 (years), and the structure of blocks H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50.
   - Read the `Data` sheet rows 21-38 to understand the lookup source — print all values to see how series codes and years are arranged (which row/column has codes, which has years, etc.).
   - Print the exact cell references to understand 1-indexed vs 0-indexed column/row mapping in openpyxl.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each yellow cell in these three blocks, write a spreadsheet formula (not a Python computation — an actual Excel formula string) that looks up data from the `Data` sheet rows 21:38.

The formula must use TWO inputs:
- The series code from column D of the current row on the `Task` sheet
- The year from row 10 of the current column on the `Task` sheet

Use one of these allowed patterns: `INDEX/MATCH`, `VLOOKUP/MATCH`, `HLOOKUP/MATCH`, or `XLOOKUP/MATCH`.

To determine the correct formula pattern, you MUST first inspect the `Data` sheet layout:
- Determine if the data table is arranged with series codes in a column and years in a row (or vice versa).
- Identify the exact range for the lookup array, match arrays, and data body.
- Choose INDEX/MATCH/MATCH if the data is a 2D table with series codes in one dimension and years in another.

For example, if the Data sheet has series codes in column A (rows 21:38) and years in a header row, an INDEX/MATCH/MATCH formula like:
`=INDEX(Data!$B$21:$F$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$F$20,0))`
would be appropriate — but you MUST adjust ranges based on what you actually observe.

Make sure:
- Column D reference is row-absolute but column-relative (e.g., `$D12`) so it varies by row.
- Row 10 reference is column-absolute for the row but varies by column (e.g., `H$10`).
- Data sheet ranges use absolute references (`$`).
- Apply the formula to every cell in all three blocks (H12:L17, H19:L24, H26:L31).

## Step 2: Net container flow in H35:L40 and statistics in H42:L47

For H35:L40, write Excel formulas for Net container flow:
`= (Loaded_Inbound - Loaded_Outbound) / Terminal_Throughput_Capacity * 100`

Based on the block structure:
- H12:L17 is likely one metric (e.g., Loaded Containers Inbound)
- H19:L24 is likely another metric (e.g., Loaded Containers Outbound)
- H26:L31 is likely Terminal Throughput Capacity

Verify which block corresponds to which metric by reading column D series codes and matching them to the Data sheet. The formula for cell H35 would be something like: `=(H12-H19)/H26*100` — adjust row offsets so each port's row in 35:40 maps to the corresponding port rows in the three blocks above.

For H42:L47, write column-wise aggregate formulas over H35:L40:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

Verify the order of these statistics by reading any labels in columns A-G for rows 42-47. Match the formula to the label.

## Step 3: Weighted mean in H50:L50

For each column H through L, write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This computes the weighted mean of Net container flow percentages weighted by Terminal Throughput Capacity.

## Final Steps
1. Do NOT create new sheets, add macros, VBA, external links, or helper tabs.
2. Do NOT change any existing formatting.
3. Save the workbook to `/root/output/result.xlsx`.
4. After saving, reopen the file and print the formula cells to verify formulas were written correctly (spot-check a few cells from each block).
5. Verify the file exists and is non-empty.

## Critical Reminders
- Write FORMULA STRINGS to cells, not computed Python values.
- Inspect before writing — understand the exact layout of both sheets.
- Use openpyxl and write formulas as strings starting with `=`.
- Keep all existing content and formatting intact.

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