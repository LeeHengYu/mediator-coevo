# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl (with `data_only=False` so formulas are preserved). Inspect:
- Sheet names (confirm `Task` and `Data` exist).
- On sheet `Task`: read row 10 to see the year headers in columns H–L. Read column D rows 12–17, 19–24, 26–31 to see the series codes. Read row 11 or nearby to understand what each block represents (e.g., Committed Funding, Operating Spend, Approved Budget Base). Read rows 35–40 to see department names/labels. Read rows 42–47 to see stat labels (min, max, median, mean, 25th, 75th percentile). Read row 50 for the Campus Budget Council label.
- On sheet `Data`: read rows 21–38 to understand the data layout — identify which column holds series codes, which row holds years, and how the data is arranged (rows vs columns). Also check row 20 or nearby header rows.

Print all of this information clearly before proceeding.

## 2. Determine the lookup formula pattern
Based on the inspection:
- Identify whether Data!rows 21–38 are arranged with series codes in a column and years across columns, or vice versa.
- Choose the appropriate lookup pattern. A good default is `INDEX(MATCH, MATCH)` which works regardless of orientation.
- The formula in each yellow cell (e.g., H12) should use two inputs: the series code from column D of that row on Task sheet, and the year from row 10 of the Task sheet.
- Construct the formula template. For example, if Data has series codes in column A rows 21–38 and years in row 20 columns B onward:
  `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
  Adjust the exact ranges based on what you find in the inspection.

## 3. Populate yellow cells in H12:L17, H19:L24, H26:L31 (Step 1)
Using openpyxl, write the lookup formula into each cell in these three blocks. Make sure:
- The series code reference uses an absolute column (`$D12`) so it doesn't shift horizontally.
- The year reference uses an absolute row (`H$10`) so it doesn't shift vertically.
- The data ranges on the Data sheet are fully absolute (e.g., `Data!$A$21:$A$38`).
- Each cell gets the correct formula (iterate row by row, column by column).

## 4. Populate H35:L40 — Net budget buffer (Step 2, part 1)
The formula is: `(Committed Funding - Operating Spend) / Approved Budget Base * 100`

Based on the block layout from inspection:
- Committed Funding block is likely H12:L17
- Operating Spend block is likely H19:L24  
- Approved Budget Base block is likely H26:L31

Confirm which block is which from the labels in the Task sheet (check column C or nearby for block titles around rows 11, 18, 25).

For each cell in H35:L40, write a formula like:
`=(H12-H19)/H26*100`
(adjusting row references to match the corresponding department row in each block)

Specifically, if the six departments in rows 35–40 correspond to the six rows in each block:
- Row 35 uses rows 12, 19, 26
- Row 36 uses rows 13, 20, 27
- Row 37 uses rows 14, 21, 28
- Row 38 uses rows 15, 22, 29
- Row 39 uses rows 16, 23, 30
- Row 40 uses rows 17, 24, 31

Verify the department order matches between the blocks and rows 35–40.

## 5. Populate H42:L47 — Summary statistics (Step 2, part 2)
For each column H through L:
- H42 (MIN): `=MIN(H35:H40)`
- H43 (MAX): `=MAX(H35:H40)`
- H44 (MEDIAN): `=MEDIAN(H35:H40)`
- H45 (AVERAGE): `=AVERAGE(H35:H40)`
- H46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- H47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Check the labels in column C/D/E/F/G around rows 42–47 to confirm the exact order of these statistics. Assign formulas accordingly.

## 6. Populate H50:L50 — Weighted mean (Step 3)
For each column H through L, write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This computes the weighted mean of the Net budget buffer percentages (H35:H40) weighted by Approved Budget Base (H26:H31).

## 7. Save and validate
- Save the workbook to `/root/output/result.xlsx` preserving all existing formatting.
- Reopen the saved file and verify:
  - All cells in H12:L17, H19:L24, H26:L31 contain formulas (not values) with lookup patterns.
  - All cells in H35:L40 contain the budget buffer formula.
  - All cells in H42:L47 contain statistical formulas.
  - All cells in H50:L50 contain SUMPRODUCT formulas.
  - No extra sheets were added.
  - Print a sample of formulas from each block to confirm correctness.

## Important constraints
- Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify existing formatting (fonts, colors, borders, etc.). Only write formulas into the specified cells.
- Use `openpyxl` with `data_only=False` when loading to preserve any existing formulas elsewhere.
- When saving, do not change the file format.

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