# Task Instruction

Execute the following steps in order to produce `/root/output/result.xlsx`.

## 0 — Inspect the workbook
```
cp /root/data/workbook.xlsx /root/output/result.xlsx
```
Open `/root/output/result.xlsx` with openpyxl and print:
- Sheet names.
- The contents of sheet `Task` rows 10-50, columns D-L (values AND any existing formulas). Pay special attention to:
  - Row 10 (years row)
  - Column D rows 12-17, 19-24, 26-31 (series codes)
  - The labels in column A/B/C for rows 35-50
- Sheet `Data` rows 21-38 fully (all columns) — identify the layout: which row holds which series code, which column holds which year, and where the actual data values sit.

Print everything clearly so the next steps can be precise.

## 1 — Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each yellow cell in those ranges, write an Excel formula that:
- Uses the series code from column D of that row on sheet `Task`.
- Uses the year from row 10 of the same column on sheet `Task`.
- Looks up the value from sheet `Data` rows 21:38.
- Uses one of the allowed patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH.

Choose whichever pattern best fits the Data layout. The most common robust choice is `INDEX(MATCH,MATCH)` — use it unless the layout clearly favors another.

IMPORTANT: After inspecting the Data sheet layout, construct the formula so that:
- One MATCH finds the row by matching the series code in column D against the series-code column on the Data sheet (rows 21-38).
- The other MATCH finds the column by matching the year in row 10 against the year header row on the Data sheet.
- INDEX returns the intersection.

Lock references appropriately (mixed references with $ where needed) so the formula can be placed across the H-L and row ranges correctly. Write the formulas using openpyxl by setting `cell.value = '=FORMULA...'` (string starting with `=`).

## 2 — Net capacity headroom (H35:L40)

For each of the six hospital clusters (rows 35-40) and each year column (H-L):
```
= (Available_Care_Slots - Occupied_Care_Slots) / Staffed_Bed_Capacity * 100
```
where:
- Available Care Slots = the corresponding cell in H12:L17
- Occupied Care Slots = the corresponding cell in H19:L24
- Staffed Bed Capacity = the corresponding cell in H26:L31

So for cell H35: `=(H12-H19)/H26*100`, H36: `=(H13-H20)/H27*100`, etc.

Verify that the row offsets are consistent: row 35 corresponds to row 12, 19, 26; row 36 to 13, 20, 27; etc.

## 3 — Summary statistics (H42:L47)

For each year column (H through L), in rows 42-47 place:
- Row 42 (MIN):  `=MIN(H35:H40)`
- Row 43 (MAX):  `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

Confirm the labels in column A/B/C match this ordering. If the labels say a different order (e.g., row 42 is MAX not MIN), adjust accordingly.

## 4 — Weighted mean (H50:L50)

For each year column col in {H,I,J,K,L}:
```
=SUMPRODUCT(col35:col40, col26:col31) / SUM(col26:col31)
```
This computes the weighted mean of the Net capacity headroom percentages weighted by Staffed Bed Capacity.

## 5 — Save and validate

Save the workbook to `/root/output/result.xlsx` (it should already be at that path).

Then reopen it with openpyxl (data_only=False) and print:
- All formulas in H12:L17, H19:L24, H26:L31 (spot-check a few)
- All formulas in H35:L40
- All formulas in H42:L47
- All formulas in H50:L50

Confirm no sheets were added, no macros, no external links. Confirm the file exists at `/root/output/result.xlsx`.

## Critical Notes
- Do NOT use data_only=True when writing; open normally with openpyxl.
- Do NOT delete or modify any existing formatting, merged cells, or content outside the target ranges.
- Formulas must be Excel-compatible strings starting with `=`.
- If the Data sheet uses a different structure than expected (e.g., series codes in a row instead of a column), adapt the INDEX/MATCH orientation accordingly — but document what you found.
- If any cell in the target ranges already has content, overwrite it with the correct formula.
- Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) for maximum compatibility unless the labels specifically indicate otherwise.

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