# Task Instruction

Complete the following spreadsheet task. Work carefully and inspect the workbook structure before writing any formulas.

## Setup
1. Create the output directory: `mkdir -p /root/output`
2. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx` so you work on the copy.
3. Install `openpyxl` if needed: `pip install openpyxl`

## Inspection Phase (Critical)
Write a Python script to inspect the workbook thoroughly before making any changes:
- Read sheet names and confirm `Task` and `Data` exist.
- On sheet `Task`: print rows 10-50 for columns A through L (or at least D through L). Pay special attention to:
  - Row 10: the year headers in columns H through L
  - Column D rows 12-17: series codes for the first block
  - Column D rows 19-24: series codes for the second block
  - Column D rows 26-31: series codes for the third block
  - Rows 35-40: which ports correspond to Net container flow
  - Rows 42-47: labels for min, max, median, mean, 25th percentile, 75th percentile
  - Row 50: the CPA weighted mean row
  - Also check columns A-G for any labels, port names, weight references, etc.
- On sheet `Data`: print rows 21-38 to understand the lookup source structure. Print the header row(s) too (likely row 20 or row 1) to understand column layout. Print enough columns to see the full data range.
- Identify the exact column layout of the Data sheet: which column has series codes, which columns have year data, etc.

## Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas
Using openpyxl, write spreadsheet formulas (not computed values) into the yellow cells. Each formula must use one of these patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.

- The two lookup keys are: (a) the series code from column D of the current row, and (b) the year from row 10 of the current column.
- The lookup source is sheet `Data` rows 21:38.
- Based on your inspection, determine the correct references for the Data range. Use appropriate absolute references ($) so formulas work across the H:L columns and down the rows.
- Use INDEX+MATCH+MATCH as the most robust two-dimensional lookup, or VLOOKUP with MATCH for the column index. Choose whichever fits the data layout.
- IMPORTANT: When writing formulas referencing the Data sheet, use the syntax `Data!A21:Z38` (or whatever the actual range is). Make sure the sheet reference is correct.

## Step 2: Net container flow in H35:L40 and statistics in H42:L47

### H35:L40 - Net container flow
The formula is: `(Loaded Containers Inbound - Loaded Containers Outbound) / Terminal Throughput Capacity * 100`
- From your inspection, identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to Loaded Containers Inbound, Loaded Containers Outbound, and Terminal Throughput Capacity.
- Write cell formulas (not Python-computed values) that reference the Step 1 cells. For example, if row 12 is port 1's inbound, row 19 is port 1's outbound, and row 26 is port 1's capacity, then H35 = (H12 - H19) / H26 * 100. Adjust based on actual layout.

### H42:L47 - Column-wise statistics
Write formulas for each column (H through L):
- Minimum: `=MIN(H35:H40)`
- Maximum: `=MAX(H35:H40)`
- Median: `=MEDIAN(H35:H40)`
- Simple mean: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`
Match the row labels from your inspection to the correct statistic.

## Step 3: Weighted mean in H50:L50
Use SUMPRODUCT for each column. The formula computes a weighted mean of the Net container flow percentages (H35:H40) weighted by Terminal Throughput Capacity (H26:H31):
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
Adjust column letters for each of H through L.

## Important Rules
- Write Excel **formulas as strings** in openpyxl (e.g., `cell.value = '=INDEX(...)'`). Do NOT compute values in Python.
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting (fonts, colors, borders, etc.).
- After writing all formulas, save to `/root/output/result.xlsx`.
- After saving, re-open the file and print out the formula content of a sample of cells (e.g., H12, H19, H26, H35, H42, H50) to verify they contain proper formula strings.

## Validation
- Verify all target cells contain formula strings starting with '='.
- Verify no existing content outside the target ranges was modified.
- Verify the file opens without error.

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