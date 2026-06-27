# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
cp /root/data/workbook.xlsx /root/output/result.xlsx
```

## 1. Inspect the workbook structure
Open `/root/output/result.xlsx` with openpyxl and inspect:
- Sheet names (confirm `Task` and `Data` exist).
- On sheet `Task`: read row 10 (the year headers in columns H–L), column D rows 12–17, 19–24, 26–31 to understand the series codes, and rows 35–40 labels, row 42–47 labels, row 50 label.
- On sheet `Data`: read rows 21–38 to understand the data layout (column headers, row labels, data orientation). Determine whether Data is organized with years in columns or rows, and where series codes appear.
- Print all of this information so you understand the exact cell references before writing any formulas.

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write a formula that looks up data from sheet `Data` rows 21:38. The formula must use two keys:
- The series code from column D of the current row on sheet `Task`
- The year from row 10 of the current column on sheet `Task`

Use INDEX/MATCH (nested) pattern. The exact formula structure depends on the Data sheet layout you discovered in step 1. For example, if Data has series codes in a column and years in a header row:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Adjust the ranges based on what you found. Important:
- Use `$D12` (absolute column, relative row) so the series code reference stays in column D when copied across columns.
- Use `H$10` (relative column, absolute row) so the year reference stays in row 10 when copied down rows.
- Make the Data ranges absolute (e.g., `Data!$A$21:$A$38` for the lookup column).
- Write formulas as strings using openpyxl's cell.value assignment. openpyxl will preserve them as formulas if the string starts with `=`.

## 3. Populate H35:L40 — Net production slack

These cells correspond to six plants. The formula for each cell is:
```
=(cell_from_Finished_Output_block - cell_from_Scrap_And_Rework_block) / cell_from_Rated_Production_Capacity_block * 100
```

From the inspection in step 1, determine which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to:
- Finished Output
- Scrap And Rework  
- Rated Production Capacity

The block labels should be visible on the Task sheet (likely in column A or nearby, around rows 11, 18, 25). Use direct cell references (e.g., `=(H12-H19)/H26*100` if the blocks map that way). Adjust row references for each of the 6 plants.

## 4. Populate H42:L47 — Summary statistics

For each column H through L, calculate over the range of 6 values in rows 35–40:
- Row 42: Minimum → `=MIN(H35:H40)`
- Row 43: Maximum → `=MAX(H35:H40)`
- Row 44: Median → `=MEDIAN(H35:H40)`
- Row 45: Simple mean → `=AVERAGE(H35:H40)`
- Row 46: 25th percentile → `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- Row 47: 75th percentile → `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Check the row labels on the Task sheet to confirm which row is which statistic, and assign formulas accordingly. The order (min, max, median, mean, 25th, 75th) must match the actual labels in the spreadsheet.

## 5. Populate H50:L50 — Weighted mean (Regional Output Council)

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net production slack percentages (H35:H40) weighted by Rated Production Capacity (H26:H31). Adjust the capacity range if your inspection showed a different block for Rated Production Capacity.

## 6. Save and verify
- Save the workbook with openpyxl to `/root/output/result.xlsx`.
- Reopen it and verify:
  - All target cells contain formula strings (start with `=`).
  - No new sheets were added.
  - The formulas reference the correct sheets and ranges.
  - Print a sample of formulas from each block to confirm correctness.

## Critical constraints
- Do NOT use `data_only=True` when loading.
- Do NOT add any new sheets, macros, VBA, or external links.
- Do NOT alter existing formatting — only set cell values (formulas).
- When loading with openpyxl, preserve existing content by not dropping any sheets.
- Use `keep_vba=False` (default) since we must not add VBA.
- If the workbook has defined names or styles, preserve them by not stripping them.

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