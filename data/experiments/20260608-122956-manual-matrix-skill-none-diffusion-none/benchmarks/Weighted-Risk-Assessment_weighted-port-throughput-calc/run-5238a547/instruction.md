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
- On sheet `Task`: read row 10 (especially H10:L10) to see the years. Read column D rows 12-17, 19-24, 26-31 to see the series codes. Read rows 35-40 column D or similar to see port names. Read H35:L40, H42:L47, H50:L50 to confirm they are empty/yellow target cells. Read any labels in column A-G for rows 12-31, 35-50 to understand the three data blocks (Loaded Containers Inbound, Loaded Containers Outbound, Terminal Throughput Capacity) and the six ports.
- On sheet `Data`: read rows 21-38 to understand the layout — identify which column holds the series codes, which row holds years, and where the numeric data lives. Print out rows 20-38 fully (all columns up to column Z or so) so we can see headers and data.

Print all of this information before proceeding.

## 2. Populate H12:L17, H19:L24, H26:L31 with lookup formulas (Step 1)

Using openpyxl, write formulas into each cell in these three blocks. Each formula should use INDEX/MATCH (or one of the other allowed patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH). The formula in each cell must:
- Look up the series code from column D of the current row on sheet `Task`.
- Look up the year from row 10 of sheet `Task`.
- Find the matching data in `Data!` rows 21:38.

Based on your inspection of the Data sheet layout, construct the correct formula. For example, if Data sheet has series codes in column A and years across row 20 (or whichever header row), an INDEX/MATCH formula might look like:
`=INDEX(Data!$B$21:$Z$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$Z$20,0))`

Adjust the exact ranges based on what you find in the inspection step. The key requirements:
- The series code reference must use `$D` (column-absolute) so it stays on column D when copied across columns.
- The year reference must use `$10` (row-absolute) so it stays on row 10 when copied down rows.
- The Data sheet ranges must be fully absolute.
- Use the exact same formula pattern for all three blocks, just with the row numbers changing naturally.

Write the formula as a string into each cell. Make sure openpyxl does NOT have `data_only=True` (load with default so formulas are preserved).

## 3. Populate H35:L40 with Net Container Flow formulas (Step 2 - first part)

The six ports in rows 35-40 should correspond to the same six ports in the blocks above (rows 12-17, 19-24, 26-31). The formula for each cell is:
`=(H12-H19)/H26*100`
(adjusting row/column references for each cell position)

So for cell H35: `=(H12-H19)/H26*100`
For cell H36: `=(H13-H20)/H27*100`
... and so on, with columns H through L.

Verify by checking that the Inbound block is rows 12-17, Outbound is rows 19-24, and Capacity is rows 26-31. If the blocks map differently to the ports, adjust accordingly.

## 4. Populate H42:L47 with summary statistics (Step 2 - second part)

For each column H through L:
- Row 42 (Minimum): `=MIN(H35:H40)`
- Row 43 (Maximum): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Simple Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

Check the labels in column A-G for rows 42-47 to confirm the correct order (min, max, median, mean, 25th, 75th). Adjust the row assignments if the labels indicate a different order.

## 5. Populate H50:L50 with weighted mean (Step 3)

For each column H through L:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This computes the weighted mean of Net Container Flow percentages using Terminal Throughput Capacity as weights.

## 6. Save and verify

Save the workbook. Then reopen it and print out the formulas in all target cells to confirm they are correctly written. Specifically:
- Print formulas in H12:L17, H19:L24, H26:L31 (spot check a few)
- Print formulas in H35:L40
- Print formulas in H42:L47
- Print formulas in H50:L50

Also verify:
- No new sheets were added
- The file is saved at `/root/output/result.xlsx`
- Formulas use the required patterns (INDEX+MATCH or similar for lookups, SUMPRODUCT for weighted mean)

## Critical Notes
- Do NOT use `data_only=True` when loading — this would strip formulas.
- Do NOT modify any existing formatting, values, or structure.
- When writing formulas with openpyxl, the formula string must start with `=`.
- Be very careful about absolute vs relative references in formulas.
- If the Data sheet inspection reveals that the data layout differs from assumptions (e.g., years in rows instead of columns, or series codes in a different column), adapt all formulas accordingly.
- The percentile functions should use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) unless you determine the verifier expects a specific variant. `PERCENTILE` is safest for broad compatibility.

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