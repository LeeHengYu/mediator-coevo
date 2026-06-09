# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
- Open `/root/data/workbook.xlsx` with openpyxl.
- Read sheet `Task`: print the contents of rows 10-50 (columns D through L) to understand the layout — specifically:
  - Row 10: the year headers in H10:L10
  - Column D rows 12-31: the series codes
  - The yellow cell regions H12:L17, H19:L24, H26:L31
  - Rows 35-40: campus names/labels for Net renewable balance
  - Rows 42-47: labels for min, max, median, mean, 25th percentile, 75th percentile
  - Row 50: MCEC weighted mean row
- Read sheet `Data`: print rows 21-38 to understand the data layout (which row has headers, which column has series codes, where year columns are, etc.).
- Print all of this before making any edits.

## 2. Populate H12:L17, H19:L24, H26:L31 with lookup formulas (Step 1)

For each cell in these ranges, write a spreadsheet formula (not a Python value) that looks up data from the `Data` sheet rows 21:38. The formula must use two inputs:
- The series code from column D of the current row on sheet `Task`
- The year from row 10 of the current column on sheet `Task`

Use one of these patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.

IMPORTANT: You must determine the exact layout of the Data sheet (which column holds series codes, which row holds year headers, the exact range) from your inspection in step 1. Construct the formula references accordingly.

For example, if Data sheet has series codes in column A and year headers in row 21 with data in rows 22-38 columns A-F, an INDEX/MATCH formula might look like:
`=INDEX(Data!$B$22:$F$38,MATCH($D12,Data!$A$22:$A$38,0),MATCH(H$10,Data!$B$21:$F$21,0))`

Adjust the ranges based on actual inspection. The formula must be a string starting with '=' assigned to the cell. Use `ws['H12'] = '=INDEX(...)'` style assignment (do NOT use `cell.value =` with a non-formula string).

Make sure:
- Row references for the series code column use `$D12` pattern (absolute column, relative row) so the series code changes per row.
- Year references use `H$10` pattern (relative column, absolute row) so the year changes per column.
- Data sheet ranges are fully absolute (e.g., `Data!$A$22:$A$38`).

## 3. Populate H35:L40 with Net renewable balance formulas (Step 2 part 1)

From the inspection, determine which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to:
- Renewable Generation
- Grid Consumption  
- Baseline Energy Demand

The formula for each campus (rows 35-40) and each year (columns H-L) is:
`=(RenewableGeneration - GridConsumption) / BaselineEnergyDemand * 100`

For example, if Renewable Generation is in rows 12-17, Grid Consumption in rows 19-24, and Baseline Energy Demand in rows 26-31, then for H35:
`=(H12-H19)/H26*100`

Adjust row offsets so each campus row in 35-40 maps to the corresponding campus row in the three blocks (first campus = first row of each block, etc.).

## 4. Populate H42:L47 with summary statistics (Step 2 part 2)

For each column H through L, in the six rows 42-47, write formulas for:
- Row 42 (minimum): `=MIN(H35:H40)` (adjust column)
- Row 43 (maximum): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (simple mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

CHECK the labels in column D/E/F/G for rows 42-47 to confirm which row is which statistic. Assign formulas accordingly — do NOT assume the order above; use the actual labels.

## 5. Populate H50:L50 with weighted mean (Step 3)

For each column H through L:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This computes the weighted mean of the Net renewable balance percentages using Baseline Energy Demand as weights.

## 6. Save
- Save the workbook to `/root/output/result.xlsx` preserving existing formatting.
- Use `wb.save('/root/output/result.xlsx')`.

## 7. Verify
- Reopen `/root/output/result.xlsx` and print cells H12, H19, H26, H35, H42, H50 to confirm they contain formula strings (starting with '=').
- Print a sample of cells to verify formulas look correct.
- Confirm no new sheets were added.
- Confirm the file exists and is non-empty.

## Critical Reminders
- All formulas must be Excel formula strings (starting with '='), NOT computed Python values.
- Do NOT change any existing formatting, do NOT add sheets.
- Inspect before editing — read the actual cell contents and layout before writing formulas.
- After writing formulas, re-read cells to confirm they were stored correctly.
- If openpyxl has issues with formula names (e.g., XLOOKUP may not be recognized), prefer INDEX+MATCH which is universally supported.
- Use `data_only=False` when loading the workbook to preserve formula capability.

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