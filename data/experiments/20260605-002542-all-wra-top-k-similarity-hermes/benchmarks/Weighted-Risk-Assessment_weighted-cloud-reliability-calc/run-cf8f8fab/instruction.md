# Task Instruction

Complete the following task to update an Excel workbook with formulas.

## Phase 0: Inspect the workbook
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx` first, then work on the copy.
2. Open and inspect the `Task` sheet thoroughly:
   - Read the structure of rows 10-50, columns D through L. Pay special attention to:
     - Column D (series codes for each row in ranges 12:17, 19:24, 26:31, 35:40)
     - Row 10 (years in columns H through L)
     - Row labels in column A-G for rows 12-17, 19-24, 26-31 (these are three blocks of 6 regions)
     - Rows 35-40 (six regions for Net reliability gap)
     - Rows 42-47 (labels for min, max, median, mean, 25th percentile, 75th percentile)
     - Row 50 (Global Cloud Mesh weighted mean)
   - Note the exact cell references, series codes, and year values.
3. Inspect the `Data` sheet:
   - Read rows 21-38 to understand the data layout. Determine:
     - Which column contains the series codes (to match against column D on Task sheet)
     - Which row contains the year headers (to match against row 10 on Task sheet)
     - Whether the data is arranged for VLOOKUP (codes in leftmost column, years across top) or HLOOKUP or needs INDEX/MATCH.
   - Note the exact range boundaries of the data table on the Data sheet.

## Phase 1: Populate lookup formulas (Step 1)
For each cell in the three blocks `H12:L17`, `H19:L24`, and `H26:L31` on sheet `Task`:
- Write a spreadsheet formula using one of the allowed patterns: `INDEX` with `MATCH` is the most flexible and recommended.
- Each formula must use TWO inputs:
  a. The series code from column D of the SAME row on the Task sheet
  b. The year from row 10 of the SAME column on the Task sheet
- The lookup source is sheet `Data` rows 21:38.
- Use appropriate absolute/relative references so formulas can be consistent across the block. Use `$` signs to anchor the lookup ranges and the row/column references that should stay fixed.
- Example pattern (adapt based on actual Data sheet layout):
  `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
  Adjust the actual column/row references based on what you find in the Data sheet.

IMPORTANT: Before writing formulas, confirm:
- The exact column on Data sheet that holds series codes
- The exact row on Data sheet that holds year headers
- The exact data range boundaries

## Phase 2: Net reliability gap and statistics (Step 2)
For cells `H35:L40` (six regions), calculate:
`= (Successful API Requests - Failed API Requests) / Compute Capacity * 100`

Based on the three blocks from Step 1:
- Block 1 (H12:L17) likely corresponds to one metric (e.g., Successful API Requests)
- Block 2 (H19:L24) likely corresponds to another metric (e.g., Failed API Requests)
- Block 3 (H26:L31) likely corresponds to Compute Capacity
Verify which block is which by reading the row/block labels on the Task sheet.

The formula for H35 would be something like: `=(H12-H19)/H26*100` (adjust based on actual block meanings and row alignment for each region).

For `H42:L47`, calculate column-wise statistics over `H35:L40`:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)
Match the actual row labels to determine the correct order.

## Phase 3: Weighted mean (Step 3)
For `H50:L50`, use SUMPRODUCT to calculate the weighted mean:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
This uses Net reliability gap percentages as values and Compute Capacity as weights.
Adjust column references for each of H through L.

## Phase 4: Validation
1. Re-open the saved `/root/output/result.xlsx` and verify:
   - Formulas exist in all required cells (spot-check several)
   - No extra sheets were added
   - The formulas reference the correct ranges
2. Use Python openpyxl to read the file and confirm formulas are present (not just values) in the target cells.
3. Ensure the file is saved and no formatting was destroyed.

## Constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Work only inside the existing `Task` and `Data` sheets.
- Save final result to `/root/output/result.xlsx`.

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