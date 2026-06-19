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
- On `Task` sheet: read row 10 (years in H10:L10), column D rows 12-17, 19-24, 26-31 (series codes), row 35-40 labels, row 42-47 labels, row 50 label.
- On `Data` sheet: read rows 21-38 to understand the data layout (column headers, row headers, what the series codes map to, where years appear).
- Print all of this so you understand the exact cell references, series codes, year values, and data layout before writing any formulas.

## 2. Understand the Data sheet layout
Determine:
- Whether Data rows 21:38 are organized with series codes in a column and years across columns (suitable for VLOOKUP+MATCH or INDEX+MATCH), or years in rows and series in columns (suitable for HLOOKUP+MATCH).
- Which column contains the series/lookup codes, which row contains the year headers.
- Print enough cells to be certain of the layout.

## 3. Write formulas in H12:L17, H19:L24, H26:L31 (Step 1)
Using openpyxl, write Excel formulas (not computed values) into each yellow cell. For each cell at row `r`, column `c` (H=8, I=9, J=10, K=11, L=12):
- The series code is in `Task!D{r}`
- The year is in `Task!{col_letter}10` where col_letter corresponds to column c
- The data range is `Data!$A$21:$Z$38` (or whatever the actual extent is — adjust based on inspection)
- Use INDEX+MATCH+MATCH pattern, e.g.:
  `=INDEX(Data!$B$21:$Z$38,MATCH($D{r},Data!$A$21:$A$38,0),MATCH({col}10,Data!$B$20:$Z$20,0))`
  Adjust the exact ranges based on what you found in step 1-2. The key is:
  - Row lookup: MATCH the series code from column D against the series code column in Data
  - Column lookup: MATCH the year from row 10 against the year header row in Data
  - INDEX into the data body

IMPORTANT: Use absolute references for the Data ranges (with $) and mixed references for the series code ($D{r}) and year ({col}$10) so the pattern is consistent. Each cell gets its own formula string.

Make sure to set each cell's value to the formula string (starting with `=`). Do NOT use `data_only` mode.

## 4. Write formulas in H35:L40 (Step 2 - Net production slack)
For each cell at row `r` in 35-40 and column `c` in H-L:
- Identify which row contains "Finished Output" data — this should be in the H12:L17 block (rows 12-17). Determine the exact row offset: if row 35 corresponds to the first plant, it maps to row 12 for Finished Output.
- Similarly identify "Scrap And Rework" block (H19:L24, rows 19-24) and "Rated Production Capacity" block (H26:L31, rows 26-31).
- The mapping: row 35→plant 1 (rows 12,19,26), row 36→plant 2 (rows 13,20,27), etc.
- Formula: `=({col}{finished_row}-{col}{scrap_row})/{col}{capacity_row}*100`
- Example for H35: `=(H12-H19)/H26*100`

## 5. Write formulas in H42:L47 (Step 2 - Statistics)
For each column c (H through L):
- H42: `=MIN({c}35:{c}40)`
- H43: `=MAX({c}35:{c}40)`
- H44: `=MEDIAN({c}35:{c}40)`
- H45: `=AVERAGE({c}35:{c}40)`
- H46: `=PERCENTILE({c}35:{c}40,0.25)`
- H47: `=PERCENTILE({c}35:{c}40,0.75)`
Verify the order by checking row labels in column D or nearby columns for rows 42-47. Adjust the row-to-statistic mapping based on actual labels found in inspection.

## 6. Write formulas in H50:L50 (Step 3 - Weighted mean)
For each column c (H through L):
- `=SUMPRODUCT({c}35:{c}40,{c}26:{c}31)/SUM({c}26:{c}31)`
This computes the weighted mean of the Net production slack percentages (H35:L40) weighted by Rated Production Capacity (H26:L31).

## 7. Save and verify
- Save the workbook to `/root/output/result.xlsx`.
- Reopen it and verify:
  - No new sheets were added
  - Formulas exist in all target cells (spot check a few)
  - No data_only values were accidentally written
  - The workbook has exactly the sheets `Task` and `Data`

## Critical notes
- Use `openpyxl` to read and write. When reading to inspect, do NOT use `data_only=True`.
- Every target cell must contain an Excel formula string (starting with `=`), not a Python-computed value.
- Do not modify any cells outside the specified ranges.
- Do not change formatting, add sheets, or add macros.
- Before writing formulas, print the inspection results and reason about the correct ranges. If the Data sheet layout differs from assumptions, adapt accordingly.
- After writing all formulas, re-read several cells to confirm they contain formula strings.

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