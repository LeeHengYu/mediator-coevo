# Task Instruction

Execute the following steps precisely to complete the weighted campus energy balance workbook task.

## Pre-work: Inspect the workbook

1. Copy the workbook: `cp /root/data/workbook.xlsx /root/output/result.xlsx`
2. Use `openpyxl` to open `/root/output/result.xlsx` and inspect:
   - Sheet names (confirm `Task` and `Data` exist).
   - On sheet `Task`: read row 10 (years in H10:L10), column D rows 12-17, 19-24, 26-31 (series codes), rows 35-40 column D (campus names or codes), row 50 label, rows 42-47 labels (min, max, median, mean, 25th, 75th).
   - On sheet `Data`: read rows 21-38 to understand the layout — identify which row is the header row, which column holds series codes, and how data is arranged (is it a vertical table with series codes in one column and years across columns, or something else?).
   - Print all of this information so you understand the exact structure before writing any formulas.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write a spreadsheet formula (not a Python-computed value) that looks up data from sheet `Data` rows 21:38. The formula must use two keys:
- The series code from column D of the current row on sheet `Task`
- The year from row 10 of the current column on sheet `Task`

Based on the Data sheet layout you discovered, choose the appropriate pattern:

- If Data has series codes in a column and years across a header row, use `INDEX(MATCH, MATCH)` like: `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))` — adjust ranges to match the actual layout.
- Alternatively use `VLOOKUP` + `MATCH`, `HLOOKUP` + `MATCH`, or `XLOOKUP` + `MATCH`.

IMPORTANT: Use `openpyxl` to write these as string formulas (e.g., `ws['H12'] = '=INDEX(...)'`). Do NOT use `data_only` mode. Make sure:
- Row references for the Data range are absolute.
- The series code reference uses absolute column (`$D12`) so it doesn't shift when copied across columns.
- The year reference uses absolute row (`H$10`) so it doesn't shift when copied down rows.
- Verify the exact column letters and row numbers from your inspection.

## Step 2: Net renewable balance formulas in H35:L40 and summary stats in H42:L47

For H35:L40, write formulas computing:
`(Renewable Generation - Grid Consumption) / Baseline Energy Demand * 100`

Based on the block layout:
- H12:L17 is one block (likely Renewable Generation)
- H19:L24 is another block (likely Grid Consumption)  
- H26:L31 is the third block (likely Baseline Energy Demand)

Confirm which block is which by reading the labels on the Task sheet (check cells around rows 11, 18, 25 for headers). Then for each cell, e.g., H35:
`=(H12-H19)/H26*100` — adjust row offsets so campus 1 in row 35 maps to campus 1 in rows 12, 19, 26, etc.

For H42:L47, write column-wise summary formulas over H35:L40:
- Row 42: `=MIN(H$35:H$40)` (or whichever row is labeled minimum)
- Row 43: `=MAX(H$35:H$40)`
- Row 44: `=MEDIAN(H$35:H$40)`
- Row 45: `=AVERAGE(H$35:H$40)`
- Row 46: `=PERCENTILE(H$35:H$40,0.25)` (or `PERCENTILE.INC`)
- Row 47: `=PERCENTILE(H$35:H$40,0.75)`

Match the actual row labels you read from the sheet to the correct function.

## Step 3: Weighted mean in H50:L50

For each column H through L, write:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the net renewable balance percentages using Baseline Energy Demand as weights.

## Final steps

1. Save the workbook with `wb.save('/root/output/result.xlsx')`. Do NOT use `data_only=True` when opening.
2. Re-open the saved file and verify:
   - All formula cells contain string formulas (not None or numeric values from Python).
   - Print a sample of formulas from each block to confirm correctness.
   - Confirm no new sheets were added.
   - Confirm the file exists at `/root/output/result.xlsx`.

## Critical constraints
- Write Excel formulas as strings, not computed Python values.
- Do not add sheets, macros, VBA, external links, or helper tabs.
- Do not alter existing formatting (do not set fonts, fills, borders, etc.).
- Use `openpyxl` without `data_only=True`.
- Ensure all formulas reference the correct sheet name if cross-sheet (e.g., `Data!...`).
- Double-check every range reference against the actual inspected layout before writing.

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