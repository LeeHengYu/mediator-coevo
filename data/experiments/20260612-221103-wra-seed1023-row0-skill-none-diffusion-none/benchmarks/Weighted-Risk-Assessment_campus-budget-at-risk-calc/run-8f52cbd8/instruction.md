# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```
mkdir -p /root/output
cp /root/data/workbook.xlsx /root/output/result.xlsx
```

## 1. Inspect the workbook structure
Open `/root/output/result.xlsx` with openpyxl and inspect:
- Sheet names (should include `Task` and `Data`)
- On sheet `Task`: read row 10 (years in columns H through L), column D rows 12-17, 19-24, 26-31 to understand the series codes, read any labels in column A or nearby for the three blocks and for rows 35-40, 42-47, 50.
- On sheet `Data`: read rows 21-38 to understand the data layout — identify which row/column holds series codes, which holds years, and how data is arranged (is it a vertical table with series codes in one column and years across columns, or something else?).
- Print all of this information so you can construct correct formulas.

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write spreadsheet formulas (not Python-computed values) into each yellow cell. Each formula must use one of the allowed lookup patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.

Key requirements:
- Each formula references TWO inputs: (a) the series code from column D of the SAME row on sheet `Task`, and (b) the year from row 10 of the SAME column on sheet `Task`.
- The lookup range is on sheet `Data` rows 21:38.
- Use absolute references for the Data range and row 10 / column D references should be mixed references (lock row for year, lock column for series code) so the formula is correct for every cell in the block.

Example pattern using INDEX+MATCH (adjust based on actual Data layout):
- If Data has series codes in column A rows 21:38 and years in row 20 (or a header row), then:
  `=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))`
- Adjust column/row ranges based on what you actually find in the Data sheet.

IMPORTANT: Before writing formulas, confirm:
- Exactly which column on Data contains the series codes
- Exactly which row on Data contains the year headers
- The data value range boundaries
- Whether years are stored as numbers or strings (match the type used in Task row 10)

Write the formulas using openpyxl. For each cell, assign a string starting with `=` to the cell's `.value` property. Do NOT use `data_only=True` when loading.

## 3. Populate Net budget buffer in H35:L40

These cells should contain formulas (not hardcoded values). Based on the block structure:
- Row 35 corresponds to department 1, row 40 to department 6.
- The formula is: `(Committed Funding - Operating Spend) / Approved Budget Base * 100`
- From the three blocks: H12:L17 is one metric, H19:L24 is another, H26:L31 is the third. Determine which block is "Committed Funding", which is "Operating Spend", and which is "Approved Budget Base" by reading labels on the Task sheet (likely in column A or B near rows 12, 19, 26).
- For example, if block 1 (rows 12-17) = Committed Funding, block 2 (rows 19-24) = Operating Spend, block 3 (rows 26-31) = Approved Budget Base, then cell H35 = `=(H12-H19)/H26*100`.
- Adjust row references so each of the 6 department rows in 35-40 maps to the corresponding row in each block.

## 4. Summary statistics in H42:L47

For each column H through L, calculate column-wise statistics over H35:L40 (the 6 Net budget buffer values):
- Row 42: `=MIN(H35:H40)` (adjust column for each)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

IMPORTANT: Check the labels in column A/B/C for rows 42-47 to confirm the correct order (min, max, median, mean, 25th, 75th). Assign formulas in the order matching those labels.

## 5. Weighted mean in H50:L50

For each column (H through L):
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the Net budget buffer percentages as values and Approved Budget Base as weights. Confirm H26:L31 is indeed the Approved Budget Base block.

## 6. Save and validate

- Save the workbook (openpyxl save to `/root/output/result.xlsx`).
- Reopen the file and verify:
  - Sheets are still only `Task` and `Data` (no new sheets added)
  - Spot-check a few cells to confirm they contain formula strings (start with `=`)
  - Confirm cells H12, H19, H26, H35, H42, H50 all have formulas
  - Print sample formulas for verification

## Critical reminders
- Do NOT use `data_only=True` when loading — you need to write formulas, not values.
- Preserve all existing formatting: load with `openpyxl.load_workbook(path)` (default keeps formatting).
- Do NOT add any sheets, macros, VBA, or external links.
- All formulas must be Excel-compatible spreadsheet formulas stored as strings.
- Use `SUMPRODUCT` specifically for the weighted mean (Step 5) as required by the task.

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