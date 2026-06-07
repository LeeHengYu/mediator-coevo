# Task Instruction

Complete the following task to update an Excel workbook with formulas.

## Setup
1. Create `/root/output/` directory if it doesn't exist.
2. Open `/root/data/workbook.xlsx` and thoroughly inspect both sheets (`Task` and `Data`) before making any changes.
3. On the `Task` sheet, inspect:
   - Column D rows 12-31 to understand the series codes for each row.
   - Row 10 columns H-L to understand the year headers.
   - The structure of the three blocks: H12:L17, H19:L24, H26:L31 (what metric each block represents).
   - Rows 35-40 (Net renewable balance), rows 42-47 (statistics), and row 50 (weighted mean).
   - Any labels in column A-G that clarify what each block/row represents.
4. On the `Data` sheet, inspect:
   - Rows 21-38 to understand the data layout: which column holds series codes, which columns/rows hold years, and how data is organized.
   - Determine the exact column/row structure so you can build correct lookup formulas.

## Important: Use openpyxl to write formulas
Use Python with `openpyxl` to load the workbook and write Excel formula strings into cells. Do NOT compute values in Python — write actual Excel formulas as strings so they evaluate in Excel.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these ranges, write a formula that:
- Uses the series code from column D of that cell's row on the `Task` sheet.
- Uses the year from row 10 of that cell's column on the `Task` sheet.
- Looks up the corresponding value from the `Data` sheet rows 21:38.
- Uses one of these patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.

Before writing formulas, determine:
- Whether Data rows 21:38 are organized with series codes in a column and years across a row (suggesting INDEX/MATCH or HLOOKUP), or years in a column and series codes across a row (suggesting VLOOKUP).
- The exact range references needed (e.g., if series codes are in Data!A21:A38 and year headers are in Data!B20:Z20, etc.).
- Use absolute references (with $) where appropriate so formulas can be consistent across the range.

## Step 2: Net renewable balance in H35:L40 and statistics in H42:L47
For H35:L40, write formulas computing:
`(Renewable Generation - Grid Consumption) / Baseline Energy Demand * 100`

Determine which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to Renewable Generation, Grid Consumption, and Baseline Energy Demand by reading the labels on the Task sheet. Then reference the appropriate cells. The six campuses in rows 35-40 should correspond to the six campuses in the data blocks (rows 12-17, 19-24, 26-31).

For H42:L47, write column-wise statistical formulas over H35:L40:
- Row 42: MIN
- Row 43: MAX
- Row 44: MEDIAN
- Row 45: AVERAGE (simple mean)
- Row 46: PERCENTILE (25th) or PERCENTILE.INC(range, 0.25)
- Row 47: PERCENTILE (75th) or PERCENTILE.INC(range, 0.75)

Check the row labels on the Task sheet to confirm the exact order of these statistics.

## Step 3: Weighted mean in H50:L50
Write a SUMPRODUCT formula for each column H-L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
This computes the weighted mean of the Net renewable balance percentages (H35:L40) weighted by Baseline Energy Demand (H26:L31). Adjust the Baseline Energy Demand range reference if it maps to a different block.

## Final Steps
1. Do NOT alter any existing formatting, sheets, or structure.
2. Save the workbook to `/root/output/result.xlsx`.
3. Reopen the saved file and verify that formulas are present in the expected cells (spot-check a few cells in each range to confirm they contain formula strings, not None or computed values).

## Key Cautions
- Write Excel formula strings (starting with '='), not Python-computed values.
- Inspect the actual workbook structure before assuming any layout.
- Use the exact row/column references from the actual workbook — do not guess.
- Preserve all existing content and formatting.

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