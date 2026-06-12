# Task Instruction

## Task: Weighted Campus Energy Balance Calculation

You need to update `/root/data/workbook.xlsx` with spreadsheet formulas and save the result to `/root/output/result.xlsx`.

### Phase 0: Inspect the workbook
1. Create `/root/output/` directory if it doesn't exist.
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet names (should have `Task` and `Data`).
   - On sheet `Task`: read row 10 to find the years in columns H through L. Read column D rows 12-17, 19-24, 26-31 to find the series codes. Read row 35-40 column D or B to find campus names. Read rows 42-47 column D or B to find which stats go where (min, max, median, mean, 25th percentile, 75th percentile). Read row 50 to understand the MCEC weighted mean row.
   - On sheet `Data`: read rows 21-38 to understand the data layout — what's in each column, how series codes and years are arranged. Determine whether years run across columns (horizontal) or data is vertical.
3. Print all of this information so you understand the exact layout before writing any formulas.

### Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write a spreadsheet formula (not a Python-computed value) that looks up data from sheet `Data` rows 21:38. The formula must use:
- The series code from column D of the current row on `Task` sheet
- The year from row 10 of the current row's column on `Task` sheet
- One of these patterns: `INDEX/MATCH`, `VLOOKUP/MATCH`, `HLOOKUP/MATCH`, or `XLOOKUP/MATCH`

IMPORTANT: You must use `openpyxl` to write Excel formula strings (e.g., `=INDEX(Data!$B$21:$Z$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$Z$20,0))`) into each cell. Adjust the exact ranges based on what you discovered in Phase 0. The formulas should be relative/mixed references so they work correctly across the block.

Key considerations:
- Identify exactly which column on `Data` contains the series codes (the lookup key matching column D on Task).
- Identify exactly which row on `Data` contains the years (the lookup key matching row 10 on Task).
- Adjust the INDEX/MATCH ranges accordingly. Use absolute references for the data range and lookup arrays, and mixed references ($D12 for series code column, H$10 for year row) so formulas copy correctly across the 5-column × 6-row blocks.

### Phase 2: Net renewable balance in H35:L40

Write spreadsheet formulas in H35:L40 that calculate:
`(Renewable Generation - Grid Consumption) / Baseline Energy Demand * 100`

Based on the layout:
- H12:L17 likely corresponds to one of these three metrics (Renewable Generation, Grid Consumption, or Baseline Energy Demand)
- H19:L24 corresponds to another
- H26:L31 corresponds to the third

Determine which block is which by reading the labels in column B or C near rows 12, 19, 26. Then write the formula referencing the correct cells. For example, if H12:L17 = Renewable Generation, H19:L24 = Grid Consumption, H26:L31 = Baseline Energy Demand, then H35 = `=(H12-H19)/H26*100`. Adjust based on actual layout.

### Phase 3: Summary statistics in H42:L47

Write column-wise spreadsheet formulas for each column H through L:
- Minimum: `=MIN(H35:H40)` (or the equivalent range)
- Maximum: `=MAX(H35:H40)`
- Median: `=MEDIAN(H35:H40)`
- Simple mean: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Match each statistic to the correct row (42-47) based on the labels you read in Phase 0.

### Phase 4: Weighted mean in H50:L50

Write a `SUMPRODUCT`-based formula for the MCEC weighted mean:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This uses the Net renewable balance percentages (H35:H40) as values and the Baseline Energy Demand block (H26:L31) as weights. Write this for each column H through L.

### Phase 5: Save and validate
1. Save the workbook to `/root/output/result.xlsx` preserving all existing formatting.
2. Re-open the saved file and verify:
   - All cells in H12:L17, H19:L24, H26:L31 contain formula strings (start with `=`).
   - All cells in H35:L40 contain formula strings.
   - All cells in H42:L47 contain formula strings.
   - All cells in H50:L50 contain formula strings.
   - No new sheets were added.
   - Print a sample of the formulas to confirm correctness.

### Critical rules
- Do NOT compute values in Python and write them as numbers. Write Excel formula strings.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Use `openpyxl` with `data_only=False` (default) so formulas are preserved.
- When loading the workbook, do NOT use `data_only=True`.

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