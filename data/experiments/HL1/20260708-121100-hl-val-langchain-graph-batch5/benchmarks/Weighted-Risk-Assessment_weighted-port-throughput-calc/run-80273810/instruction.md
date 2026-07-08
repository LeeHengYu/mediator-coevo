# Task Instruction

You must update an Excel workbook at `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Follow these steps precisely.

## Step 0: Inspect the workbook

1. `mkdir -p /root/output`
2. Use `openpyxl` (Python) to open `/root/data/workbook.xlsx` and inspect:
   - Sheet names (should include `Task` and `Data`).
   - On sheet `Task`: read cells D12:D17, D19:D24, D26:D31 to see the series codes for each block. Read row 10 (H10:L10) to see the year headers. Read H35:H40 area and D35:D40 for port names/codes. Read any labels in column D or G near rows 42-47 and row 50. Note any existing content/formatting in the yellow target cells.
   - On sheet `Data`: read rows 21-38 to understand the data layout — identify which row is the header row, what columns contain series codes, and how years are arranged (horizontally or vertically). Print out enough to understand the structure.
3. Print all findings before proceeding.

## Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas

Each yellow cell in these three blocks needs a formula that looks up a value from `Data!` rows 21:38 using:
- The series code from column D of that row on the `Task` sheet
- The year from row 10 of the `Task` sheet

Use an `INDEX(MATCH,MATCH)` pattern (or `XLOOKUP` with `MATCH`, or `VLOOKUP` with `MATCH` — pick one pattern and use it consistently). The formula must reference:
- The data range on sheet `Data` (rows 21:38, appropriate columns)
- A MATCH on the series code against the series-code column in `Data`
- A MATCH on the year against the year header row in `Data`

IMPORTANT: Use absolute references for the data range and lookup arrays, but allow the series code reference (column D) and year reference (row 10) to vary appropriately so the formula can fill across the 5 columns (H-L) and down the 6 rows in each block. Specifically:
- The column D reference should be fixed in column ($D) but relative in row so it changes per row.
- The row 10 year reference should be fixed in row ($10) but relative in column so it changes per column.

Write the formulas using `openpyxl` by setting each cell's `.value` to a string starting with `=`. Make sure the sheet reference uses the exact sheet name (e.g., `Data!`).

After writing, re-read a few cells to confirm the formulas are stored correctly.

## Step 2: Net container flow and summary statistics in H35:L47

Based on your inspection, identify which of the three blocks corresponds to:
- Loaded Containers Inbound (likely H12:L17)
- Loaded Containers Outbound (likely H19:L24)  
- Terminal Throughput Capacity (likely H26:L31)

Verify by checking the labels near rows 11, 18, 25 on the Task sheet.

For H35:L40, write formulas: `(Inbound - Outbound) / Capacity * 100`
For example, if Inbound is in row 12 and Outbound in row 19 and Capacity in row 26, then H35 = `=(H12-H19)/H26*100`. Adjust row references for each of the 6 ports.

For H42:L47, write column-wise summary statistics over H35:L40:
- Row 42: `=MIN(H35:H40)` (adjust column for each)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

Check the labels in column D/G for rows 42-47 to confirm the correct order of min/max/median/mean/25th/75th. Adjust row assignments to match the actual labels.

## Step 3: Weighted mean in H50:L50

For each column (H through L), write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This computes the weighted mean of net container flow percentages using Terminal Throughput Capacity as weights.

## Step 4: Save and validate

1. Save the workbook to `/root/output/result.xlsx` preserving all existing formatting. When opening with openpyxl, do NOT use `data_only=True`. Use `keep_vba=False` (default). Make sure to NOT destroy existing styles — open the workbook without specifying style options that would strip formatting.
2. Re-open `/root/output/result.xlsx` and print out:
   - A sample of formulas from each block (H12, L17, H19, L24, H26, L31, H35, H40, H42, H47, H50, L50)
   - Confirm they are formula strings (start with `=`)
3. Verify no extra sheets were added.

## Critical constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify existing formatting/styles.
- Work only inside the `Task` and `Data` sheets.
- All cell references in formulas must be correct for the actual workbook layout you discover in Step 0.

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