# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
cp /root/data/workbook.xlsx /root/output/result.xlsx
```

## 1. Inspect the workbook structure
Open `/root/output/result.xlsx` with openpyxl and inspect:
- Sheet names (confirm `Task` and `Data` exist)
- On sheet `Task`: read row 10 (years in H10:L10), column D rows 12-17, 19-24, 26-31 (series codes), rows 35-40 labels, rows 42-47 labels, row 50 label
- On sheet `Data`: read rows 21-38 to understand the data layout (headers, row keys, column keys)
- Identify what the yellow cells currently contain (should be empty or placeholder)
- Print all of this information so you understand the exact structure before writing formulas

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each yellow cell in these three blocks, write a spreadsheet formula (not a Python-computed value). The formula must use one of the allowed lookup patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH.

The two inputs for every lookup are:
- The series code in column D of the current row (e.g., `$D12` for row 12)
- The year in row 10 of the current column (e.g., `H$10` for column H)

The source data is on sheet `Data` in rows 21:38. You need to understand the layout of that data range to construct the correct formula. The formula should search for the series code in the appropriate column/row of the Data sheet and the year in the appropriate row/column, then return the intersection value.

IMPORTANT: Use absolute references for the lookup ranges on the Data sheet so formulas can be filled across the block. Use mixed references ($D12 for series code column, H$10 for year row) so the formula works when placed in different cells of the block.

When writing formulas with openpyxl, just assign the formula string to the cell's value, e.g. `cell.value = '=INDEX(Data!$B$21:$Z$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$Z$20,0))'`. Adjust column/row references based on the actual Data sheet layout you discovered in step 1.

## 3. Populate Net SLA buffer formulas in H35:L40

The formula for each cell is:
`(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`

You need to identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to each of these three quantities. Look at the labels in the Task sheet (likely in column A-G area near rows 11, 18, 25) to determine which block is which.

For each cell in H35:L40, write a formula referencing the corresponding cells from those blocks. For example, if block 1 is Latency Budget Preserved (rows 12-17), block 2 is Latency Budget Consumed (rows 19-24), and block 3 is Covered Request Capacity (rows 26-31), then H35 would be `=(H12-H19)/H26*100`. Adjust row offsets so each of the 6 services maps correctly.

## 4. Populate summary statistics in H42:L47

For each column H through L, calculate column-wise statistics over the Net SLA buffer values (H35:H40 for column H, etc.):
- Row 42: MIN (e.g., `=MIN(H35:H40)`)
- Row 43: MAX (e.g., `=MAX(H35:H40)`)
- Row 44: MEDIAN (e.g., `=MEDIAN(H35:H40)`)
- Row 45: AVERAGE (e.g., `=AVERAGE(H35:H40)`)
- Row 46: PERCENTILE (25th) (e.g., `=PERCENTILE(H35:H40,0.25)`)
- Row 47: PERCENTILE (75th) (e.g., `=PERCENTILE(H35:H40,0.75)`)

IMPORTANT: Check the actual labels in column A-G for rows 42-47 to determine which row gets which statistic. Map them correctly based on the labels you read.

## 5. Populate weighted mean in H50:L50

Use SUMPRODUCT with the Net SLA buffer percentages (H35:H40 for column H) as values and the Covered Request Capacity block (H26:H31 for column H) as weights:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

Repeat for columns I through L with appropriate column references.

## 6. Save and verify

- Save the workbook (already at `/root/output/result.xlsx`)
- Re-open and verify that:
  - All formula cells contain formula strings (start with `=`)
  - No sheets were added or removed
  - The formulas reference the correct ranges
  - Print a sample of formulas from each block to confirm correctness

## Critical Notes
- Use `openpyxl` to read and write the workbook
- When loading, do NOT use `data_only=True` (you need to preserve and write formulas)
- Do NOT change any existing formatting, values, or structure
- Do NOT add new sheets, macros, VBA, external links, or helper tabs
- All formulas must be Excel spreadsheet formulas, not Python-computed values
- Read the actual workbook structure carefully before writing any formulas — the exact row/column layout of the Data sheet is essential for correct lookup formulas

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