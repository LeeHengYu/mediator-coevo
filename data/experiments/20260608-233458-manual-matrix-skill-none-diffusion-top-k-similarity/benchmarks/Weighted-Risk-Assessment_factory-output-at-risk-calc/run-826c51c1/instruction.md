# Task Instruction

Execute the following multi-phase plan to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Phase 0 – Setup
```
mkdir -p /root/output
pip install openpyxl
```

## Phase 1 – Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Print:
1. Sheet names.
2. On sheet `Task`: the contents of column D rows 12-17, 19-24, 26-31 (series codes), and row 10 columns H-L (years).
3. On sheet `Data`: rows 21-38, all populated columns – print enough to understand the layout (header row, key columns, data columns).
4. On sheet `Task`: the contents of rows 35-40 column D (plant names for Net production slack), rows 42-47 column D or E (stat labels), and row 50 columns D-G (label for weighted mean).
5. Identify which row/column on `Data` holds the series codes and which row holds years so we know the lookup axes.

Do NOT edit anything yet. Just print and study.

## Phase 2 – Populate H12:L17, H19:L24, H26:L31 with INDEX/MATCH formulas
Using the information from Phase 1, write INDEX/MATCH formulas into the yellow cells. Each formula should:
- Use the series code from column D of the same row on `Task`.
- Use the year from row 10 of the same column on `Task`.
- Look up data on sheet `Data` in the range spanning rows 21:38.
- Use absolute references for the Data lookup range and MATCH ranges so they don't shift when filled across rows/columns.
- Pattern: `=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))`

Adjust the exact range references based on what you found in Phase 1. Fill all 18 rows × 5 columns = 90 cells (H12:L17, H19:L24, H26:L31).

## Phase 3 – Net production slack (H35:L40)
For each of the six plants (rows 35-40) and each year column (H-L), write a formula:
```
=(H12 - H19) / H26 * 100
```
where H12 is the Finished Output cell, H19 is the Scrap And Rework cell, and H26 is the Rated Production Capacity cell for the same plant and year. Adjust row references per plant. Use relative/mixed references appropriately.

Verify: The three blocks 12-17, 19-24, 26-31 should each have the same six plants in the same order as rows 35-40. Confirm this from Phase 1 output before writing formulas. If the order differs, map correctly.

## Phase 4 – Summary statistics (H42:L47)
For each year column H through L, write these formulas referencing the Net production slack block H35:H40 (adjust column letter per column):
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE.INC(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE.INC(H35:H40, 0.75)`

**IMPORTANT**: Use `PERCENTILE.INC` (not `PERCENTILE`) to avoid #NAME? errors. Verify the row-label mapping from Phase 1 – the order of MIN/MAX/MEDIAN/MEAN/P25/P75 may differ from what I listed; match each formula to its actual label in column D/E.

## Phase 5 – Weighted mean (H50:L50)
For each year column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net production slack using Rated Production Capacity as weights.

Wait – re-check: the standard weighted mean formula is `SUMPRODUCT(values, weights) / SUM(weights)`. However, the instruction says to use SUMPRODUCT with the Step 2 percentages as values and the Rated Production Capacity block as weights. So the formula should be:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This is correct.

## Phase 6 – Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Reopen it and verify:
   - No new sheets were added.
   - Spot-check a few formula cells (e.g., H12, H19, H26, H35, H42, H46, H47, H50) to confirm they contain the expected formula strings.
   - Confirm PERCENTILE.INC is used (not PERCENTILE).
3. Print confirmation.

## Key constraints
- Use openpyxl throughout. Do not use xlsxwriter or other libraries.
- Do not modify formatting, do not add sheets, macros, VBA, external links, or helper tabs.
- Read the actual file content before every edit phase. Never assume cell contents from memory.
- If any formula pattern needs adjustment based on the actual layout discovered in Phase 1, adapt accordingly but keep the same lookup strategy (INDEX/MATCH for lookups, PERCENTILE.INC for percentiles, SUMPRODUCT for weighted mean).

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