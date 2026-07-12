# Task Instruction

You must update an Excel workbook by writing spreadsheet formulas (not Python-computed values) into specific cells, then save the result.

## Setup
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx` so you always work on the output copy.
2. Open `/root/output/result.xlsx` with `openpyxl` (use `load_workbook` with `keep_vba=False`).
3. **Before writing anything**, inspect the workbook thoroughly:
   - Print every sheet name.
   - On sheet `Task`: print rows 1–50 completely (all columns A–L at minimum) so you can see the layout, the yellow cells, column D series codes, row 10 years, and any existing content.
   - On sheet `Data`: print rows 1–40 completely to understand the data layout, column headers, and where series codes / years appear.
   - Pay special attention to: (a) what is in column D for rows 12–17, 19–24, 26–31 (the series codes), (b) what years are in H10:L10, (c) how the Data sheet rows 21–38 are structured (which column has the series code, which row/column has years, and where the values are).

## Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these ranges, write an Excel formula that:
- Takes the series code from column D of that row on sheet `Task`
- Takes the year from row 10 of the same column on sheet `Task`
- Looks up the value from sheet `Data` rows 21–38

Choose the appropriate lookup pattern based on the Data sheet layout. If Data has series codes in a column and years across a row, use `INDEX(MATCH, MATCH)` — e.g.:
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
Adjust the exact ranges after inspecting the Data sheet. The key references:
- The series-code column in Data (likely column A or B)
- The year header row in Data (likely row 20 or the row just above row 21)
- The data block in Data (rows 21:38, columns with year data)
- Use absolute references for the Data ranges and mixed references ($D12 for series code column, H$10 for year row) so formulas can be placed across the grid.

**Important**: Use `Translator` from openpyxl or manually construct each formula string with correct cell references. Write formulas as strings starting with `=`.

## Step 2: Net capacity headroom in H35:L40
For each of the 6 hospital clusters (rows 35–40) and each year column (H–L), write a formula:
```
=(H12 - H19) / H26 * 100
```
where row 12 = Available Care Slots, row 19 = Occupied Care Slots, row 26 = Staffed Bed Capacity — adjusted for the actual row of each cluster. Specifically:
- Row 35 uses data from rows 12, 19, 26
- Row 36 uses data from rows 13, 20, 27
- Row 37 uses data from rows 14, 21, 28
- Row 38 uses data from rows 15, 22, 29
- Row 39 uses data from rows 16, 23, 30
- Row 40 uses data from rows 17, 24, 31

## Step 2 continued: Summary statistics in H42:L47
For each column H–L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

Check the labels in column A/B/C for rows 42–47 to confirm which row is which statistic. Adjust row assignments to match the labels.

## Step 3: Weighted mean in H50:L50
For each column H–L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net capacity headroom using Staffed Bed Capacity as weights.

## Final steps
- Do NOT change any formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.
- Save the workbook to `/root/output/result.xlsx`.
- After saving, reopen and print the formula cells to verify they contain formula strings (starting with `=`), not computed values.
- Also spot-check that the Data sheet references are correct by printing a few Data cells that should be found by the lookups.

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