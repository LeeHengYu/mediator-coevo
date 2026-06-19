# Task Instruction

Execute the following steps to complete the task:

## Step 0: Setup and Inspect
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (keep formulas as-is: `data_only=False`).
3. Inspect the `Task` sheet:
   - Read row 10 to identify the years in columns H through L.
   - Read column D for rows 12–17, 19–24, 26–31 to identify the series codes for each block.
   - Read row labels in rows 35–40 (campus names for Net renewable balance), rows 42–47 (stat labels: min, max, median, mean, 25th pctl, 75th pctl), and row 50 (MCEC weighted mean).
4. Inspect the `Data` sheet:
   - Read rows 21–38 to understand the layout: which row contains the series code, which row contains the year headers, and how the data is arranged (likely series codes in a column, years across a row, or vice versa).
   - Identify the exact column that holds the series codes and the exact row that holds the year headers on the Data sheet. Note these precisely (e.g., column A for series codes, row 20 for years).

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write an INDEX/MATCH formula that:
- Uses the series code from column D of the current row.
- Uses the year from row 10 of the current column.
- Looks up data from the `Data` sheet rows 21:38.

The exact formula pattern depends on the Data sheet layout found in Step 0. A typical pattern if Data has series codes in a column (say column A) and years in a row (say row 20):
```
=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```
Adjust the ranges based on what you actually find. The key contract: use INDEX with MATCH (or one of the other approved patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH).

Write the formula strings into cells using openpyxl (e.g., `ws['H12'] = '=INDEX(...)'`). Do NOT use data_only or cached values—write actual formula strings.

## Step 2: Net renewable balance in H35:L40
For each campus (rows 35–40) and each year column (H–L), write a formula:
```
=(H12_block_renewable - H19_block_grid) / H26_block_baseline * 100
```
where the renewable generation row corresponds to the same campus in H12:L17, grid consumption in H19:L24, and baseline energy demand in H26:L31. Map each campus row in 35–40 to the corresponding row offset in the three blocks above (row 35 → rows 12, 19, 26; row 36 → rows 13, 20, 27; etc.).

Example for H35: `=(H12-H19)/H26*100`

## Step 3: Summary statistics in H42:L47
For each column H–L, write formulas in rows 42–47. Read the actual labels in column D/E/F/G of rows 42–47 to determine the order. Typical mapping:
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- MEAN (simple): `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)`

Match each formula to the actual label in the row. Do not assume the order—read the labels first.

## Step 4: Weighted mean in H50:L50
For each column H–L, write a SUMPRODUCT formula:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net renewable balance percentages using Baseline Energy Demand as weights.

## Step 5: Save
1. Save the workbook to `/root/output/result.xlsx` using `wb.save('/root/output/result.xlsx')`.
2. After saving, reopen the file and verify that cells H12, H35, H42, and H50 contain non-None formula strings (not empty).
3. Print confirmation of the formulas found in those cells.

## Critical Reminders
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify existing formatting.
- Ensure `wb.save()` is called—this was a failure mode in a similar task.
- Use openpyxl in non-data-only mode so formulas are preserved.
- Write formulas as strings starting with '='.
- After every major step, print what was written to confirm correctness before moving on.

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