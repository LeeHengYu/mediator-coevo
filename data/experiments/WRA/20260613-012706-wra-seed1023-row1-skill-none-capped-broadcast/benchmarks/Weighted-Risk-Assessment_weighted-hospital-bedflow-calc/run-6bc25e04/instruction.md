# Task Instruction

Execute the following steps carefully and in order.

## Phase 0 – Inspect the workbook

1. Copy the source workbook:
   ```bash
   cp /root/data/workbook.xlsx /root/output/result.xlsx
   ```
2. Open `/root/output/result.xlsx` with openpyxl (data_only=False) and inspect:
   - **Sheet `Data`**: Print rows 20-40 (all columns up to column 15) so you can see the exact layout: which row is the header, where the year values are, where the series codes are, and where the numeric data lives. Pay special attention to:
     - Which row contains the year headers (e.g., 2019, 2020, …)
     - Which column contains the series codes
     - The exact range of the data block (rows 21:38)
   - **Sheet `Task`**: Print rows 1-55, columns A-L, so you can see:
     - Row 10: the year values in columns H through L
     - Column D rows 12-17, 19-24, 26-31: the series codes
     - The labels for the three blocks and the statistics block
     - Any existing content in the yellow target cells
   - Print the exact cell values (not just a summary). This is critical.

## Phase 1 – Determine the correct formula references

From the inspection, determine:
- `DATA_SHEET_NAME`: the exact name of the Data sheet (likely `Data`)
- The column on the Data sheet that holds the series codes (e.g., column A → column index 1)
- The row on the Data sheet that holds the year headers (e.g., row 21 or row 20)
- The data range on the Data sheet: the top-left and bottom-right of the numeric block

For each yellow cell at position (row, col) on the Task sheet, you need a formula that:
- Looks up the series code from column D of the same row on Task
- Matches the year from row 10 of the same column on Task
- Returns the corresponding value from the Data sheet

Use `INDEX(MATCH,MATCH)` pattern because it is the most reliable:
```
=INDEX(Data!$B$22:$F$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$F$21, 0))
```
But adjust the exact ranges based on your inspection. The key is:
- The INDEX range covers only the numeric data cells (not the header row or code column)
- The first MATCH searches the series code column (same rows as the INDEX range)
- The second MATCH searches the year header row (same columns as the INDEX range)

**IMPORTANT**: Use `$` signs correctly: lock column D reference with `$D`, lock row 10 reference with `$10`, and lock all Data sheet ranges with absolute references.

## Phase 2 – Write lookup formulas

Using openpyxl, write formulas to cells H12:L17, H19:L24, and H26:L31 on the Task sheet. Each cell gets an INDEX/MATCH/MATCH formula as determined above.

Loop over the three blocks. For each cell (r, c) in the target range:
```python
cell.value = '=INDEX(Data!$B$22:$F$38,MATCH($D{r},Data!$A$22:$A$38,0),MATCH({col_letter}$10,Data!$B$21:$F$21,0))'
```
Adjust the range references based on your Phase 0 findings.

## Phase 3 – Write Net Patient Flow formulas (H35:L40)

For each hospital row i (0..5), the formula in cell (35+i, col) should be:
```
=(H12-H19)/H26*100
```
where H12 corresponds to the Patient Admissions block row, H19 to Patient Discharges, H26 to Effective Bed Capacity. Adjust row references:
- Admissions: rows 12-17
- Discharges: rows 19-24  
- Bed Capacity: rows 26-31

So for row 35, col H: `=(H12-H19)/H26*100`
For row 36, col H: `=(H13-H20)/H27*100`
etc.

Use relative column references so they shift across H-L, but build the formula string with the correct row numbers for each hospital.

## Phase 4 – Write statistics formulas (H42:L47)

For each column (H through L):
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC` — the verifier may not recognize the dotted versions. Actually, check: standard Excel accepts both, but openpyxl and some evaluators only handle `PERCENTILE`. Use `PERCENTILE` to be safe.

Wait — from the avoid artifact, `#NAME?` errors occurred with statistics functions. This likely means dotted function names like `PERCENTILE.INC` caused issues. Use the non-dotted versions: `PERCENTILE`, `MEDIAN`, `MIN`, `MAX`, `AVERAGE`.

## Phase 5 – Write weighted mean formula (H50:L50)

For each column (H through L):
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean of Net Patient Flow using Effective Bed Capacity as weights.

## Phase 6 – Save and verify

1. Save the workbook to `/root/output/result.xlsx`
2. Re-open the saved file with openpyxl (data_only=False) and print the formula content of a few sample cells (e.g., H12, H19, H26, H35, H42, H46, H50) to confirm formulas were written correctly.
3. Run the test suite:
   ```bash
   cd /root && python -m pytest tests/ -v 2>&1 | head -80
   ```
4. If tests fail, read the error messages carefully. Common issues:
   - Wrong range references → re-inspect Data sheet and fix
   - `#NAME?` errors → function name issue, switch to compatible names
   - `None` values → formulas not written or wrong cell addresses
   - Wrong numeric values → range offset error, re-check Phase 0 findings
5. Fix and re-run until tests pass.

## Critical Reminders
- Do NOT use `PERCENTILE.INC`, `PERCENTILE.EXC`, or any dotted function names.
- Do NOT add new sheets, macros, or VBA.
- Do NOT change existing formatting.
- The formulas must use standard Excel function names that the verifier's evaluation engine can compute.
- Double-check every range reference against the actual Data sheet layout from Phase 0.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=easy, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.