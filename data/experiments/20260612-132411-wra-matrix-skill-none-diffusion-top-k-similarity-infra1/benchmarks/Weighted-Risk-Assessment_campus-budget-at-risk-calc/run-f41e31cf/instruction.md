# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
- Open `/root/data/workbook.xlsx` with openpyxl (data_only=False to preserve formulas).
- Print the sheet names to confirm `Task` and `Data` exist.
- On sheet `Task`:
  - Print rows 10-50, columns D through L, to understand the layout: row 10 should contain years in H10:L10; column D should contain series codes; rows 12-17, 19-24, 26-31 are the three lookup blocks; row 35-40 is Net budget buffer; rows 42-47 are summary stats; row 50 is weighted mean.
  - Print the exact cell values for D12:D17, D19:D24, D26:D31, D35:D40 (series codes / department labels).
  - Print H10:L10 (the year headers).
  - Note the exact yellow-highlighted cell ranges to confirm they match H12:L17, H19:L24, H26:L31.
- On sheet `Data`:
  - Print rows 21-38 completely (all columns) to see the lookup source structure. Identify which column holds the series code and which row/column holds the year, and where the values are.
  - Determine the data layout: Is it organized with series codes in one column and years across columns, or some other arrangement? Note the exact column letters and row numbers.

## 2. Determine the lookup pattern
Based on the Data sheet structure from step 1:
- If series codes are in a column and years are in a header row within Data!rows 21-38, then INDEX/MATCH with MATCH is the cleanest approach.
- The formula pattern for each cell (e.g., H12) should use:
  - The series code from column D of the same row on Task sheet (e.g., $D12)
  - The year from row 10 of the same column on Task sheet (e.g., H$10)
  - Look these up in the Data sheet range rows 21:38

## 3. Write formulas using openpyxl
Use a Python script with openpyxl to write formulas. Important: openpyxl writes Excel formula strings (without computing them). Use `ws['H12'] = '=INDEX(...)'` style.

For each cell in H12:L17, H19:L24, H26:L31:
- Construct an INDEX/MATCH formula like:
  `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
  **BUT** adjust the exact column/row references based on what you discovered in step 1. The key references to get right:
  - The data value range on Data sheet (the block of numbers)
  - The lookup column for series codes on Data sheet
  - The lookup row for years on Data sheet
  - The series code reference on Task sheet ($D fixed column, row varies)
  - The year reference on Task sheet (column varies, $10 fixed row)

## 4. Step 2 - Net budget buffer formulas in H35:L40
The three blocks are:
- H12:L17 = one metric (likely Committed Funding based on series codes)
- H19:L24 = another metric (likely Operating Spend)
- H26:L31 = another metric (likely Approved Budget Base)

Verify which block corresponds to which metric by checking the labels near rows 11, 18, 25 on the Task sheet.

Then for H35 (first department, first year):
`=(H12 - H19) / H26 * 100`
(Adjust row references if the block assignments differ. The formula is: (Committed Funding - Operating Spend) / Approved Budget Base * 100)

Fill H35:L40 with the corresponding formulas for all 6 departments × 5 years.

## 5. Step 2 - Summary statistics in H42:L47
For each column (H through L):
- H42: `=MIN(H35:H40)`
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)`
- H46: `=PERCENTILE(H35:H40, 0.25)`
- H47: `=PERCENTILE(H35:H40, 0.75)`

Verify the order (min, max, median, mean, 25th, 75th) by checking any labels in column D or nearby for rows 42-47. Adjust the order to match whatever labels are present.

## 6. Step 3 - Weighted mean in H50:L50
For each column (H through L):
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net budget buffer percentages using Approved Budget Base as weights.

## 7. Save
- Save the workbook to `/root/output/result.xlsx`.
- Do NOT change any existing formatting, do NOT add sheets.
- Verify the saved file by reopening it and printing a sample of the formula cells to confirm formulas are written correctly.

## Critical checks before finishing
1. Re-read several cells (e.g., H12, L17, H35, L40, H42, H47, H50, L50) from the saved file to confirm formulas are present and reference the correct sheets/ranges.
2. Confirm no new sheets were added.
3. Confirm the file exists at `/root/output/result.xlsx`.

## Important notes
- Use `load_workbook(filename, data_only=False)` to preserve existing formulas.
- When writing formulas, they must start with `=`.
- Use absolute references ($) appropriately: fix the column for series code lookups ($D), fix the row for year lookups ($10), and fix the Data sheet ranges entirely.
- The lookup formulas must use one of the approved patterns: INDEX with MATCH is recommended.
- Do NOT use data_only=True as that would strip existing formulas.
- Do NOT add macros, VBA, external links, or helper tabs.

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