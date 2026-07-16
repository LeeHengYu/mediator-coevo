# Task Instruction

You must update the Excel workbook at `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Follow these steps precisely.

## Preliminary Inspection

1. Read the file `/root/data/workbook.xlsx` using openpyxl. Inspect:
   - Sheet names (confirm `Task` and `Data` exist).
   - On sheet `Task`: read row 10 to find the years in columns H through L. Read column D rows 12-17, 19-24, 26-31 to find the series codes. Read rows 35-40 column D for the port names used in Step 2. Read row 50 for the label (should mention CPA). Read rows 42-47 column D/E/F/G for the stat labels (min, max, median, mean, 25th, 75th percentile).
   - On sheet `Data`: inspect rows 21-38 to understand the data layout — which row holds which series code, and which columns hold which years. Determine the orientation: are series codes in a column and years across columns, or vice versa? Identify the exact column that contains the series codes and the exact row that contains the year headers.
2. Print all of this information so you can construct correct formulas.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on your inspection, write spreadsheet formulas (not Python calculations) into each yellow cell. Each formula must use one of these patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.

- The two lookup keys are: (a) the series code from column D of the current row on sheet `Task`, and (b) the year from row 10 on sheet `Task`.
- The lookup range is on sheet `Data` rows 21:38.
- Make sure to use correct absolute/relative references so that when you write formulas for each cell, they reference the correct series code and year.
- Use `INDEX(MATCH, MATCH)` as the preferred pattern since it handles 2D lookups cleanly. The formula pattern should be something like: `=INDEX(Data!<data_range>, MATCH(<series_code_ref>, Data!<series_code_column>, 0), MATCH(<year_ref>, Data!<year_row>, 0))`
- Adjust the exact ranges based on your inspection of the Data sheet layout.

## Step 2: Net container flow and statistics in H35:L40 and H42:L47

For H35:L40, write formulas that compute:
`(Loaded Containers Inbound - Loaded Containers Outbound) / Terminal Throughput Capacity * 100`

- Identify which rows in the Task sheet hold "Loaded Containers Inbound" (should be H12:L17), "Loaded Containers Outbound" (should be H19:L24), and "Terminal Throughput Capacity" (should be H26:L31).
- For each port (rows 35-40) and each year (columns H-L), the formula should reference the corresponding cell from each block. E.g., H35 = (H12 - H19) / H26 * 100.

For H42:L47, write column-wise aggregate formulas over H35:L40:
- Identify which row is which statistic from your inspection of column D/E labels in rows 42-47.
- MIN: `=MIN(H35:H40)` (and similarly for each column)
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- MEAN (simple): `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- 75th percentile: `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`
- Match each statistic to the correct row based on the labels you read.

## Step 3: Weighted mean in H50:L50

Write a SUMPRODUCT-based formula for the CPA weighted mean:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
(and similarly for columns I through L)

This uses the Net container flow percentages as values and Terminal Throughput Capacity as weights.

## Saving

- Create `/root/output/` directory if it doesn't exist.
- Save the workbook to `/root/output/result.xlsx`.
- Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting (do not modify fonts, fills, borders, number formats, etc.).

## Validation

After saving, re-open `/root/output/result.xlsx` and:
1. Confirm cells H12, L17, H19, L24, H26, L31 contain formulas (not bare values).
2. Confirm cells H35, L40 contain formulas.
3. Confirm cells H42, L47 contain formulas.
4. Confirm cells H50, L50 contain formulas.
5. Print a sample of formulas to verify correctness.

IMPORTANT: Use openpyxl to write formulas as strings (e.g., `ws['H12'] = '=INDEX(...)'`). Do NOT use data_only mode when writing. When reading back for validation, do NOT use data_only mode either — just confirm the formula strings are present.

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