# Task Instruction

Execute the following steps exactly to produce /root/output/result.xlsx.

## 0 – Preparation
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1 – Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Examine:
- Sheet `Task`: look at columns A–G in rows 10–50 to understand layout (series codes in column D, years in row 10, campus names, section headers).
- Sheet `Data`: look at rows 21–38 to understand the source data layout (what is in row 21 as headers, what columns hold what).
- Note the exact column letters and row numbers for series codes, years, and data ranges.

Print out enough to understand the structure before writing any formulas.

## 2 – Write lookup formulas in H12:L17, H19:L24, H26:L31

For every cell in these three blocks, write a formula that looks up the value from `Data!$A$21:$XX$38` (adjust range to actual data extent) using the series code from column D of the current row and the year from row 10 of the current column.

Use this pattern (INDEX-MATCH is safest):
```
=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```
Adjust the actual range references after inspecting the Data sheet layout. The key requirements:
- Row lookup: MATCH the series code in column D of the current row against the series-code column on Data sheet.
- Column lookup: MATCH the year in row 10 against the year header row on Data sheet.
- Use absolute references for the data range and relative references for the series code ($D12) and year (H$10) so formulas copy correctly across the 5-column × 6-row blocks.

Write formulas cell-by-cell or in a loop. Do NOT paste values; paste formula strings.

## 3 – Net renewable balance in H35:L40

For each campus (6 rows) and each year (5 columns), compute:
```
=(H12 - H19) / H26 * 100
```
where H12 is the Renewable Generation cell, H19 is the Grid Consumption cell, and H26 is the Baseline Energy Demand cell for the same campus and year. Adjust row references per campus. These should be cell-reference formulas, not hardcoded values.

## 4 – Summary statistics in H42:L47

For each year column (H through L):
- H42: `=MIN(H35:H40)`
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)`
- H46: `=PERCENTILE(H35:H40,0.25)`   ← Use `PERCENTILE` NOT `PERCENTILE.INC`
- H47: `=PERCENTILE(H35:H40,0.75)`   ← Use `PERCENTILE` NOT `PERCENTILE.INC`

**CRITICAL**: The previous run failed because of #NAME? errors on rows 46-47. The function name `PERCENTILE.INC` may cause issues. Use the plain `PERCENTILE` function name (no dot-suffix). Verify after writing that the formula string is exactly `PERCENTILE(...)` with no typos.

## 5 – Weighted mean in H50:L50

For each year column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the net renewable balance percentages using Baseline Energy Demand as weights.

## 6 – Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.

## 7 – Validate
Reopen the saved file with openpyxl (data_only=False) and print:
- A sample formula from each block (H12, H19, H26, H35, H42, H46, H47, H50)
- Confirm no cells in the target ranges are None or empty
- Confirm H46 and H47 formulas contain the string 'PERCENTILE' and do NOT contain 'PERCENTILE.INC' or 'PERCENTILE.EXC'

Also try opening with data_only=True and print values from a few cells to see if they resolve (they may show None in openpyxl without a calc engine, which is fine).

Report any issues found during validation.

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