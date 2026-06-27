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
- On sheet `Task`: read row 10 (especially H10:L10) to see the year headers; read column D rows 12-17, 19-24, 26-31 to see the series codes; read rows 35-40 to understand the service layout; read row 50 for the weighted mean row; read rows 42-47 for the stat labels.
- On sheet `Data`: read rows 21-38 to understand the data layout — identify what's in row 21 (likely headers with years), and what column A contains (series codes). Determine the column range and row range precisely.
- Print all of this information so you understand the exact cell references.

## 2. Populate H12:L17, H19:L24, H26:L31 with lookup formulas

For each yellow cell in these three blocks, write a formula that:
- Uses the series code from column D of that row
- Uses the year from row 10 of that column
- Looks up the value from `Data!$A$21:$XX$38` (adjust column range based on inspection)

Use `INDEX/MATCH` pattern. The formula pattern for cell e.g. H12 should be something like:
```
=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$21:$XX$21, 0))
```
Adjust the exact ranges based on what you find in the inspection step. The key points:
- The row match should look up `$D12` (column D of current row, with $ on D to lock column) against the series code column in Data (likely column A).
- The column match should look up `H$10` (row 10 of current column, with $ on 10 to lock row) against the header row in Data (likely row 21).
- Use absolute references for the Data ranges and mixed references for D column and row 10.

IMPORTANT: Use openpyxl to write these formulas as strings. Make sure to set `workbook.calculation.calcMode` or similar if needed, but primarily just write the formula strings into cells.

When writing formulas with openpyxl, assign the formula string directly to the cell value, e.g.:
```python
ws['H12'] = '=INDEX(Data!$B$21:$Z$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$21:$Z$21,0))'
```

## 3. Populate H35:L40 with Net SLA Buffer formula

Net SLA buffer = (Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100

Based on the three blocks:
- H12:L17 = first block (check which metric this is)
- H19:L24 = second block (check which metric)
- H26:L31 = third block (check which metric)

Inspect the labels to determine which block is "Latency Budget Preserved", "Latency Budget Consumed", and "Covered Request Capacity". The rows 35-40 correspond to the six services (same order as rows 12-17, 19-24, 26-31).

For example, if block 1 (rows 12-17) is Preserved, block 2 (rows 19-24) is Consumed, block 3 (rows 26-31) is Capacity, then for H35:
```
=(H12-H19)/H26*100
```
Adjust based on actual inspection.

## 4. Populate H42:L47 with statistics

For each column H through L, calculate over the 6 values in rows 35-40:
- Row 42: MIN, e.g. `=MIN(H35:H40)`
- Row 43: MAX, e.g. `=MAX(H35:H40)`
- Row 44: MEDIAN, e.g. `=MEDIAN(H35:H40)`
- Row 45: AVERAGE, e.g. `=AVERAGE(H35:H40)`
- Row 46: 25th percentile, e.g. `=PERCENTILE(H35:H40,0.25)`
- Row 47: 75th percentile, e.g. `=PERCENTILE(H35:H40,0.75)`

Check the labels in column D/E/F/G for rows 42-47 to confirm which row is which statistic and adjust accordingly.

## 5. Populate H50:L50 with weighted mean

Use SUMPRODUCT with the Net SLA Buffer values (H35:H40 for column H) as values and Covered Request Capacity (H26:H31 for column H) as weights:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```

## 6. Save

Save the workbook. Do NOT change any formatting, do not add sheets, macros, VBA, or external links.

## 7. Verify

Re-open the saved file and print out the formula in a few sample cells (e.g., H12, L17, H35, H42, H50) to confirm they were written correctly. Also verify sheet names are still just `Task` and `Data`.

## Critical Notes
- When opening with openpyxl, do NOT use `data_only=True` — you need to write formulas.
- Preserve all existing formatting by not touching any cells other than the ones specified.
- The exact Data range boundaries must come from your inspection in step 1. Do not guess.
- When writing formulas, make sure the formula starts with `=`.
- Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) unless inspection of existing formulas suggests otherwise. Actually, prefer `PERCENTILE.INC` as it's the modern equivalent and more commonly expected. Check if there are any existing formulas in the workbook that use a specific variant.
- Double-check that the row-to-statistic mapping (min/max/median/mean/p25/p75) matches the labels in the workbook.

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