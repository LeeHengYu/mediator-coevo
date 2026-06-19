# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
cp /root/data/workbook.xlsx /root/output/result.xlsx
```

## 1. Inspect the workbook structure
Open `/root/output/result.xlsx` with openpyxl and inspect:
- Sheet names (should be `Task` and `Data`)
- On sheet `Task`: read row 10 (especially H10:L10) to see the years; read column D rows 12-31 to see series codes; read H35:L40 area to understand layout; read rows 42-47 labels; read row 50 label; note any existing content or formatting in the yellow cells.
- On sheet `Data`: read rows 21-38 to understand the data layout (headers, series codes, years, values). Determine whether data is arranged with series codes in a column and years across columns, or vice versa.

Print all of this information clearly before proceeding.

## 2. Determine the lookup structure
Based on the Data sheet layout:
- Identify which row/column contains the series codes (lookup keys)
- Identify which row/column contains the year headers
- Identify the data range
- Decide on the best lookup pattern. The instruction allows VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.

## 3. Populate H12:L17, H19:L24, H26:L31 with lookup formulas
For each cell in these ranges, write a spreadsheet formula (not a Python computation) that:
- Uses the series code from column D of that row
- Uses the year from row 10 of that column
- Looks up the value from Data!$21:$38 (or the appropriate range)
- Uses one of the allowed lookup patterns consistently

Use openpyxl to write these formulas as strings (e.g., `ws['H12'] = '=INDEX(Data!...,...)'`). Make sure:
- References to the series code column and year row are appropriately absolute/relative
- The Data sheet range reference is correct
- The formula pattern is consistent across all cells

IMPORTANT: When writing formulas with openpyxl, do NOT set `data_only=True` when loading. Load the workbook normally so formulas are preserved.

## 4. Populate H35:L40 with Net Production Slack formulas
The formula is: `(Finished Output - Scrap And Rework) / Rated Production Capacity * 100`

Based on the inspection:
- Identify which rows in H12:L17 correspond to "Finished Output" (or similar)
- Identify which rows in H12:L17 or H19:L24 correspond to "Scrap And Rework" (or similar)
- Identify which rows in H26:L31 correspond to "Rated Production Capacity" (or similar)
- The six plants should map row-by-row

For each cell in H35:L40, write a formula like: `=(Hxx-Hyy)/Hzz*100` where xx, yy, zz are the appropriate rows for that plant and that column.

Note: Carefully check the row labels in column D (or nearby) for rows 12-17, 19-24, 26-31, and 35-40 to understand which series maps to which concept.

## 5. Populate H42:L47 with summary statistics
For each column H through L, calculate over the range of that column in rows 35:40:
- Minimum: `=MIN(H35:H40)`
- Maximum: `=MAX(H35:H40)`
- Median: `=MEDIAN(H35:H40)`
- Simple mean: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Match each formula to the correct row based on the labels in column D (or nearby) for rows 42-47. Read those labels first.

## 6. Populate H50:L50 with weighted mean using SUMPRODUCT
For each column H through L:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This computes the weighted mean of the Net Production Slack percentages (H35:H40) weighted by the Rated Production Capacity values (H26:H31).

## 7. Save and verify
- Save the workbook
- Re-open it and verify that all target cells contain formula strings (not None or plain values)
- Print a sample of formulas from each section to confirm correctness
- Confirm no new sheets were added
- Confirm the file is saved at `/root/output/result.xlsx`

## Critical Notes
- Load the workbook WITHOUT `data_only=True` so formulas are written and preserved
- Do NOT add any new sheets, macros, or VBA
- Preserve all existing formatting (do not clear cells or modify styles)
- When writing formulas, ensure they start with `=`
- Use `PERCENTILE.INC` if `PERCENTILE` causes issues (both are standard Excel functions)
- Double-check row/column mappings by reading actual cell contents before writing formulas

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