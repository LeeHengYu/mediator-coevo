# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## Preparation
1. `mkdir -p /root/output`
2. Install openpyxl if not already available: `pip install openpyxl`
3. Open and inspect `/root/data/workbook.xlsx` with openpyxl to understand the layout:
   - Read sheet `Task`: print row 10 (years header), column D rows 12-31 (series codes), row labels in column A/B for rows 35-50, and the yellow target ranges.
   - Read sheet `Data`: print rows 21-38 to understand the lookup source table structure (row headers, column headers).
   - Print any existing content in the target cells to confirm they are empty.

## Step 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three 6×5 blocks, write an Excel formula using INDEX/MATCH/MATCH that:
- Looks up the series code from column D of the current row against the Data sheet's row headers
- Looks up the year from row 10 of the Task sheet against the Data sheet's column headers
- Returns the intersection value from the Data table (rows 21:38)

Concretely, for cell H12 the formula pattern should be:
```
=INDEX(Data!$A$21:$ZZ$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$A$21:$ZZ$21,0))
```
Adjust the Data range references based on the actual extent of the Data sheet (inspect first). Use mixed references: `$D12` (column-absolute, row-relative) and `H$10` (column-relative, row-absolute) so the formula copies correctly across the 5 columns and 6 rows of each block.

Write the formula to all 90 cells (3 blocks × 6 rows × 5 columns).

## Step 2 – Net renewable balance (H35:L40) and statistics (H42:L47)
For H35:L40, write formulas computing:
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to Renewable Generation (rows 12-17), H19 to Grid Consumption (rows 19-24), and H26 to Baseline Energy Demand (rows 26-31). Adjust row references for each of the 6 campus rows, keeping column relative.

For H42:L47, write column-wise statistical formulas referencing H35:H40 (adjusting column for each):
- Row 42: `=MIN(H$35:H$40)`
- Row 43: `=MAX(H$35:H$40)`
- Row 44: `=MEDIAN(H$35:H$40)`
- Row 45: `=AVERAGE(H$35:H$40)`
- Row 46: `=PERCENTILE(H$35:H$40,0.25)`
- Row 47: `=PERCENTILE(H$35:H$40,0.75)`

IMPORTANT: Verify which row label corresponds to which statistic by inspecting column A/B of rows 42-47 before writing. Map the function to the correct label (e.g., if row 42 says "Minimum" use MIN, if it says "25th percentile" use PERCENTILE with 0.25, etc.).

## Step 3 – Weighted mean in H50:L50
For each column H through L, write:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean of the net renewable balance percentages using Baseline Energy Demand as weights.

## Final Steps
1. Save the workbook to `/root/output/result.xlsx` using openpyxl.
2. Reopen the saved file and verify:
   - Cells H12, L17, H19, L24, H26, L31 contain formula strings (not None).
   - Cells H35, L40 contain formula strings.
   - Cells H42, L47 contain formula strings.
   - Cell H50 and L50 contain formula strings.
3. Print a sample of formulas from each block to confirm correctness.

## Critical Notes
- Do NOT evaluate formulas in Python; write them as Excel formula strings.
- Do NOT modify any existing formatting, sheet names, or structure.
- Do NOT add new sheets, macros, or VBA.
- Before writing any formula, inspect the actual Data sheet to determine the exact header row and column range for the lookup table. Adjust INDEX/MATCH ranges accordingly.
- The row-to-statistic mapping in rows 42-47 MUST match the actual labels in the workbook. Inspect before writing.

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