# Task Instruction

Complete the following task to update an Excel workbook with formulas.

## Setup
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Work on `/root/output/result.xlsx` throughout.
3. First, inspect the workbook thoroughly: read sheet names, then read the `Task` sheet (especially columns A-L, rows 1-55) and the `Data` sheet (especially rows 1-40, focusing on rows 21-38) to understand the layout, series codes, years, and data structure.

## Inspection Checklist (do this BEFORE writing any formulas)
- On `Task` sheet: identify what is in row 10 (years), column D rows 12-17, 19-24, 26-31 (series codes), and the existing content/labels in column A-G for those row blocks.
- On `Data` sheet: identify the structure of rows 21-38 — what column contains series codes, what row/column contains years, and how the data is organized (is it a vertical table with series codes in one column and years across columns, or something else?).
- Identify which cells in H12:L17, H19:L24, H26:L31 are the yellow target cells.
- Identify what labels are in rows 35-40 (the six plants), rows 42-47 (statistics labels), and row 50.
- Print all findings before proceeding.

## Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write a formula that:
- Takes the series code from column D of that row (e.g., D12 for row 12)
- Takes the year from row 10 of that column (e.g., H10 for column H)
- Looks up the corresponding value from the `Data` sheet rows 21:38

Use INDEX/MATCH or one of the other allowed patterns (VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH). The exact formula depends on the Data sheet layout you discover during inspection. Use appropriate absolute/mixed references so the formula can be filled across the block correctly.

IMPORTANT: Lock references appropriately. The series code column reference (D) should be absolute on the column ($D), and the year row reference (row 10) should be absolute on the row ($10 or row 10). The lookup ranges on the Data sheet should be fully absolute.

## Step 2: Net Production Slack in H35:L40 and Statistics in H42:L47
In H35:L40, for each of the six plants, calculate:
`(Finished Output - Scrap And Rework) / Rated Production Capacity * 100`

Based on inspection, determine which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to 'Finished Output', 'Scrap And Rework', and 'Rated Production Capacity'. Use cell references to those blocks (e.g., `(H12-H19)/H26*100` or similar, adjusted to actual row mapping).

In H42:L47, calculate column-wise statistics over H35:L40:
- MIN
- MAX
- MEDIAN
- AVERAGE (simple mean)
- PERCENTILE (or PERCENTILE.INC) with 0.25 for 25th percentile
- PERCENTILE (or PERCENTILE.INC) with 0.75 for 75th percentile

Match each statistic to the correct row based on the labels you find in column A-G for rows 42-47.

## Step 3: Weighted Mean in H50:L50
For each column H through L, use:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
or equivalently a SUMPRODUCT-based weighted average formula. The values are the Step 2 percentages (H35:H40) and the weights are the Rated Production Capacity block (H26:L31). Adjust row references based on actual layout.

## Final Checks
- Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Verify the file saves correctly to `/root/output/result.xlsx`.
- Open and re-read a few cells to confirm formulas are present and return numeric values (not errors).

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