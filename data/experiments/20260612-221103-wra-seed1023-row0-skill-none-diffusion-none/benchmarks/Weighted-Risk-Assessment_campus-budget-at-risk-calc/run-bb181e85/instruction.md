# Task Instruction

Execute the following steps to populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx.

## Step 0 — Inspect the workbook
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False).
3. Print sheet names.
4. On sheet `Task`:
   - Print rows 10-11 (headers/years row) columns A-L to identify the year row (expected: row 10 has years in H-L).
   - Print rows 12-17 columns A-L to see block 1 labels and series codes in column D.
   - Print rows 19-24 columns A-L to see block 2 labels and series codes in column D.
   - Print rows 26-31 columns A-L to see block 3 labels and series codes in column D.
   - Print rows 33-50 columns A-L to see the derived-value area, stat rows, and weighted-mean row.
5. On sheet `Data`:
   - Print row 21 columns A-L to see the year header row.
   - Print rows 22-38 columns A-D to see series codes and a sample of data.
   - Identify the data range: rows and columns containing the numeric values.

Record: (a) which row on `Data` holds years, (b) which column holds series codes, (c) the top-left and bottom-right of the numeric data block on `Data`, (d) the exact column letters for years on both sheets.

## Step 1 — Lookup formulas (H12:L17, H19:L24, H26:L31)
Using the inspection results, write INDEX/MATCH formulas into every cell in these three blocks. The pattern for cell HXX should be:

```
=INDEX(Data!<data_range>, MATCH($D{row}, Data!<series_code_column>, 0), MATCH({col}$10, Data!<year_row>, 0))
```

Use mixed references so $D{row} locks the column and {col}$10 locks the row. Iterate over rows 12-17, 19-24, 26-31 and columns H-L. Adjust the `Data!` ranges based on inspection (e.g., `Data!$B$22:$F$38` for values, `Data!$A$22:$A$38` for series codes, `Data!$B$21:$F$21` for years — but verify from inspection).

## Step 2 — Net budget buffer (H35:L40) and statistics (H42:L47)
For each cell in H35:L40, write a formula:
```
=({committed_funding_cell} - {operating_spend_cell}) / {approved_budget_base_cell} * 100
```
where committed_funding is block 1 (rows 12-17), operating_spend is block 2 (rows 19-24), approved_budget_base is block 3 (rows 26-31). The row offset between the blocks should be consistent (e.g., H35 uses H12, H19, H26; H36 uses H13, H20, H27; etc.). Verify the mapping from inspection.

For rows 42-47, write column-wise statistics over H35:H40 through L35:L40. Check the labels in column A/B to determine the order. Expected (verify from labels):
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- AVERAGE (simple mean): `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40, 0.25)`
- 75th percentile: `=PERCENTILE(H35:H40, 0.75)`

Map each stat to the correct row based on the actual labels found in inspection.

## Step 3 — Weighted mean (H50:L50)
For each column H-L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the net budget buffer percentages as values and Approved Budget Base as weights.

## Step 4 — Save
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and verify:
   - Cell H12 is not None (has a formula string starting with '=').
   - Cell H35 is not None.
   - Cell H42 is not None.
   - Cell H50 is not None.
3. Print these cells' values to confirm.

## Critical Reminders
- Do NOT add sheets, macros, VBA, or helper tabs.
- Do NOT change existing formatting.
- The file MUST be saved to `/root/output/result.xlsx` — confirm the save actually happens.
- Use `wb.save(...)` explicitly after all formula writes.
- Re-read cells after writing to confirm they are not None.
- If any inspection result is unexpected, adapt the ranges accordingly rather than blindly using assumed ranges.

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