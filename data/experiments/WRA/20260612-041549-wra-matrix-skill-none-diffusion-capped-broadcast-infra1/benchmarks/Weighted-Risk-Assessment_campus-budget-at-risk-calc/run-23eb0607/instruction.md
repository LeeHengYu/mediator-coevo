# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Inspect the workbook

1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Open `/root/output/result.xlsx` with openpyxl (with `data_only=False` so formulas are preserved).
3. Inspect sheet `Task`:
   - Print rows 10–50, columns A–L (or at least D–L) to understand the layout: headers, series codes in column D, years in row 10, and which cells are currently empty (the yellow target cells).
   - Specifically note:
     a. The series codes in D12:D17, D19:D24, D26:D31, D35:D40.
     b. The years in H10:L10.
     c. Any labels in column A–C or rows 10, 18, 25, 34, 41, 48, 49 that clarify block meanings.
     d. The labels in A42:G47 (min, max, median, mean, 25th, 75th percentile).
4. Inspect sheet `Data`:
   - Print rows 20–40 to see the structure: which row has headers, where series codes and year columns are, orientation (rows vs columns).
   - Determine whether the data table is arranged with series codes in a column and years across columns, or vice versa. This determines which lookup pattern to use.

## 1. Populate H12:L17, H19:L24, H26:L31 with lookup formulas

Based on inspection, write formulas into each cell in these three blocks. Each formula must use one of the allowed patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH. The two inputs are:
- The series code from column D of the same row (e.g., $D12 for row 12).
- The year from row 10 of the same column (e.g., H$10 for column H).

The lookup range is `Data!` rows 21:38. Construct the formula so that:
- One MATCH finds the position of the series code within the appropriate axis of the data range.
- The other MATCH (or the main lookup function) uses the year.
- Use absolute references on the data range and mixed references ($D12 for series code, H$10 for year) so the formula can be applied across the 5×6 block correctly.

For example, if Data has series codes in column A rows 21:38 and years across row 20 (or a header row), an INDEX-MATCH formula might look like:
`=INDEX(Data!$B$21:$Z$38, MATCH($D12,Data!$A$21:$A$38,0), MATCH(H$10,Data!$B$20:$Z$20,0))`
Adjust column/row references based on actual inspection.

Write the formula into every cell in H12:L17, H19:L24, H26:L31 (that's 3 blocks × 6 rows × 5 columns = 90 cells).

## 2. Populate H35:L40 with Net budget buffer formulas

The three blocks are (based on typical ordering — confirm via inspection):
- Block 1 (H12:L17): one of Committed Funding, Operating Spend, or Approved Budget Base
- Block 2 (H19:L24): another
- Block 3 (H26:L31): another

Identify which block is which from the row labels (likely in column A or nearby). Then for each cell in H35:L40:
`= (Committed_Funding_cell - Operating_Spend_cell) / Approved_Budget_Base_cell * 100`

For example, if Block 1 = Committed Funding (rows 12-17), Block 2 = Operating Spend (rows 19-24), Block 3 = Approved Budget Base (rows 26-31):
`H35 = (H12 - H19) / H26 * 100`
`H36 = (H13 - H20) / H27 * 100`
etc., through L40.

Ensure the department ordering in rows 35-40 matches the ordering in the source blocks (check D35:D40 vs D12:D17).

## 3. Populate H42:L47 with summary statistics

For each column (H through L), write these formulas referencing the Net budget buffer block (e.g., H35:H40):
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- AVERAGE (simple mean): `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- 75th percentile: `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`

Match each row (42–47) to the correct statistic based on the labels in column A/G. If labels say rows 42=min, 43=max, 44=median, 45=mean, 46=25th, 47=75th, use that order. Confirm by reading the actual labels.

## 4. Populate H50:L50 with weighted mean (SUMPRODUCT)

For each column (H through L):
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net budget buffer percentages using Approved Budget Base as weights.

## 5. Save and verify

1. Save the workbook to `/root/output/result.xlsx`.
2. Reopen it and verify:
   - Cells H12:L31 contain formulas (not hardcoded values).
   - Cells H35:L40 contain formulas.
   - Cells H42:L47 contain formulas.
   - Cells H50:L50 contain SUMPRODUCT formulas.
   - No new sheets were added.
   - Print a sample of cells to confirm formulas look correct.

## Important constraints
- Use openpyxl to read and write the xlsx file.
- Do NOT use data_only=True when writing (that would strip formulas).
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Preserve all existing formatting — do not clear or overwrite cells outside the target ranges.
- All formulas must be Excel-compatible spreadsheet formulas (not Python calculations).
- Make sure to use the exact sheet name references (e.g., `Data!` prefix) in lookup formulas.

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