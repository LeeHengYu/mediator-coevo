# Task Instruction

You must update the Excel workbook at `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Work only inside the existing `Task` and `Data` sheets. Do not add sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

## Preliminary Investigation

1. First, inspect the `Task` sheet thoroughly:
   - Read the layout of rows 10-50, columns D through L.
   - Identify what is in column D (series codes for each service row).
   - Identify what is in row 10 (years for columns H through L).
   - Identify the labels/structure in rows 12-17, 19-24, 26-31 (three blocks of 6 services each).
   - Identify what rows 35-40 are labeled as, and rows 42-47 (stats), and row 50.
   - Note which cells are already populated and which are empty (the yellow target cells).

2. Inspect the `Data` sheet:
   - Read rows 21-38 carefully to understand the data layout.
   - Determine the structure: which column holds series codes, which row holds years, and where the numeric data lives.
   - Identify exactly how to look up a value given (series_code, year).

3. Read `/tests/test_outputs.py` (or any test file in `/tests/`) to understand the verifier's expectations: expected cell values, tolerances, and which cells are checked.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three 6×5 blocks, write a spreadsheet formula (not a hardcoded value) that looks up data from the `Data` sheet rows 21:38. Each formula must use:
- The series code from column D of the current row on `Task`.
- The year from row 10 of the current column on `Task`.
- One of these patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH.

Use absolute references where appropriate (e.g., lock row 10 for the year, lock column D for the series code).

IMPORTANT: When referencing the Data sheet range, make sure your range covers rows 21:38 correctly and that the MATCH dimensions align (row vs column). Verify that the series codes in column D on Task match exactly the codes in the Data sheet (check for whitespace, case, etc.).

Use openpyxl to write formulas as strings into cells (e.g., `ws['H12'] = '=INDEX(...)'`). Do NOT use data_only mode when loading.

## Step 2: Net SLA buffer in H35:L40 and statistics in H42:L47

For H35:L40, write formulas computing:
`(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`

You need to determine which of the three blocks (rows 12-17, 19-24, 26-31) corresponds to "Latency Budget Preserved", "Latency Budget Consumed", and "Covered Request Capacity". Inspect the Task sheet labels carefully.

For each cell in H35:L40, reference the corresponding cells from those blocks. For example, if block 1 is Latency Budget Preserved (rows 12-17), block 2 is Latency Budget Consumed (rows 19-24), and block 3 is Covered Request Capacity (rows 26-31), then H35 = (H12 - H19) / H26 * 100. Adjust based on actual labels.

For H42:L47, compute column-wise statistics over H35:L40 (6 values per column):
- Row 42: MIN
- Row 43: MAX  
- Row 44: MEDIAN
- Row 45: AVERAGE
- Row 46: 25th percentile
- Row 47: 75th percentile

Check the Task sheet labels to confirm the exact order of these statistics.

CRITICAL (from cross-task failures): For percentiles, use `PERCENTILE` or `PERCENTILE.INC` — do NOT use `PERCENTILE.EXC` or any non-standard function name. The #NAME? errors in similar tasks came from using unrecognized function names. Verify that `PERCENTILE.INC(range, 0.25)` and `PERCENTILE.INC(range, 0.75)` are valid, or use `PERCENTILE(range, 0.25)`. Test which one openpyxl and Excel accept.

## Step 3: Weighted mean in H50:L50

For each column H through L, write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net SLA buffer percentages (H35:H40) weighted by Covered Request Capacity (H26:H31). Adjust row references if the blocks are in different positions.

## Implementation approach

Use a Python script with openpyxl:
1. `mkdir -p /root/output`
2. Load workbook: `wb = openpyxl.load_workbook('/root/data/workbook.xlsx')` (no data_only).
3. Write all formulas as strings into the appropriate cells.
4. Save to `/root/output/result.xlsx`.
5. After saving, re-open the file and print the formulas in key cells to verify they were written correctly.

Do NOT overwrite any existing cell content outside the target ranges. Do NOT change formatting, sheet names, or structure.

Before writing any formulas, print out:
- Column D values for rows 12-17, 19-24, 26-31, 35-40, 42-47, 50
- Row 10 values for columns H-L
- The Data sheet structure (rows 21-38, first few columns and the header row)
- Labels in column A or B or C that identify each block

Use this information to construct correct formulas.

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