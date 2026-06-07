# Task Instruction

Complete the following task to populate formulas in an Excel workbook.

## Setup
1. Create the output directory: `mkdir -p /root/output`
2. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx` so you work on the copy.
3. Install openpyxl if needed: `pip install openpyxl`

## Inspection (Critical — do this first)
Write a Python script to inspect the workbook thoroughly:
- Read sheet `Task`: print the contents of rows 1–55, columns A–L (especially column D for series codes, row 10 for years, and the yellow cell ranges).
- Read sheet `Data`: print rows 1–40, all columns, to understand the data layout (especially rows 21:38 which are the lookup source).
- Note the exact series codes in column D for rows 12–17, 19–24, 26–31, and 35–40.
- Note the exact years in H10:L10.
- Note any existing formulas or values already present.
- Note the structure of the Data sheet rows 21:38 — identify which column holds the series code and which row/column holds the year headers.

## Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31
For each cell in these ranges, write a spreadsheet formula (not a computed value) that:
- Uses the series code from column D of that row and the year from row 10 of that column.
- Looks up the value from sheet `Data` rows 21:38.
- Uses one of the allowed patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH.
- Choose the pattern that best fits the Data sheet layout. If data is arranged with series codes in one column and years across columns, INDEX(MATCH,MATCH) is likely cleanest.
- IMPORTANT: Write actual Excel formula strings (starting with '=') into the cells. Do NOT compute values in Python.
- Make sure references to the Data sheet use the correct sheet name syntax, e.g., `Data!A21:A38` or similar.
- Use absolute references where appropriate (e.g., anchor the lookup range, anchor the year row reference, anchor the series code column reference) so formulas can be filled across the range correctly.

## Step 2: Net Patient Flow in H35:L40 and Statistics in H42:L47
- In H35:L40, write formulas for each hospital (6 hospitals, 5 years):
  `= (Patient Admissions - Patient Discharges) / Effective Bed Capacity * 100`
  where Patient Admissions come from H12:L17, Patient Discharges from H19:L24, and Effective Bed Capacity from H26:L31. Use cell references to these ranges (same row offset within each block, same column).
- In H42:L47, write column-wise statistical formulas over H35:L40:
  - Row 42: MIN of H35:H40 (for each column H through L)
  - Row 43: MAX
  - Row 44: MEDIAN
  - Row 45: AVERAGE (simple mean)
  - Row 46: PERCENTILE (25th) — use PERCENTILE(H35:H40, 0.25) or PERCENTILE.INC
  - Row 47: PERCENTILE (75th) — use PERCENTILE(H35:H40, 0.75) or PERCENTILE.INC
  - IMPORTANT: Check the labels in column D or nearby columns for rows 42–47 to confirm which statistic goes in which row. Match the order to whatever labels are already there.

## Step 3: Weighted Mean in H50:L50
- For each column H through L, write a SUMPRODUCT formula:
  `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)` (adjust column letter per column).
  This computes the weighted mean of Net Patient Flow using Effective Bed Capacity as weights.

## Final Checks
- After writing all formulas, re-read the workbook and verify:
  - Cells in H12:L17, H19:L24, H26:L31 contain formula strings (not bare values).
  - Cells in H35:L40 contain formula strings.
  - Cells in H42:L47 contain formula strings.
  - Cells in H50:L50 contain formula strings using SUMPRODUCT.
- Confirm no new sheets were added.
- Confirm the file is saved at `/root/output/result.xlsx`.

## Important Notes
- Use openpyxl to write formulas. When you assign a string starting with '=' to a cell's value, openpyxl stores it as a formula.
- Do NOT use data_only mode when reading for inspection (you want to see existing formulas).
- Preserve all existing formatting — do not clear or overwrite cells outside the specified ranges.
- Adapt the exact formula pattern based on what you discover during inspection (e.g., the exact range in the Data sheet, exact column/row layout).

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