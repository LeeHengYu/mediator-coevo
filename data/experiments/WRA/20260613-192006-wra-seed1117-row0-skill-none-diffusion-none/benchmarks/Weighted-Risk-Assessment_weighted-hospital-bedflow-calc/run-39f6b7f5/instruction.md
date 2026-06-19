# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
cp /root/data/workbook.xlsx /root/output/result.xlsx
```

## 1. Inspect the workbook structure
Open `/root/output/result.xlsx` with openpyxl and inspect:
- Sheet names (confirm `Task` and `Data` exist)
- On sheet `Task`: read row 10 (years in H10:L10), column D rows 12-31 (series codes), rows 35-50 layout, any existing content/formatting in the yellow cell ranges
- On sheet `Data`: read rows 21-38 to understand the data layout (what's in each column/row, where series codes and years appear, how the data is organized — is it a vertical table with series codes in one column and years across columns, or something else?)

Print all of this information before proceeding. Understanding the exact layout is critical.

## 2. Populate H12:L17, H19:L24, H26:L31 with lookup formulas

Based on the inspection, write formulas into each cell in these three blocks. Each formula should look up a value from the `Data` sheet rows 21:38 using:
- The series code from column D of the same row on `Task`
- The year from row 10 of the same column on `Task`

Use an INDEX/MATCH pattern. The exact references depend on the Data sheet layout discovered in step 1. For example, if Data has series codes in column A and years in a header row, a typical formula for cell H12 would be:
```
=INDEX(Data!$B$21:$Z$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$Z$20,0))
```
Adjust the exact ranges based on what you find in the Data sheet. The key constraints:
- Row anchor: `$D12` (column D of current row, row varies)
- Column anchor: `H$10` (row 10 of current column, column varies)
- The Data range for lookup must cover rows 21:38 on sheet Data
- Use absolute references for the data range and lookup arrays, mixed references for the two inputs so formulas copy correctly across the 5×6 grid in each block

Write formulas for ALL cells: H12:L17 (6 rows × 5 cols = 30 cells), H19:L24 (30 cells), H26:L31 (30 cells). Total 90 formula cells.

## 3. Populate H35:L40 with Net Patient Flow formulas

For each of the 6 hospitals (rows 35-40) and 5 years (columns H-L):
```
= (Admissions - Discharges) / Effective_Bed_Capacity * 100
```
where:
- Admissions are in H12:L17 (row 12 corresponds to row 35, row 13 to row 36, etc.)
- Discharges are in H19:L24 (row 19 corresponds to row 35, row 20 to row 36, etc.)
- Effective Bed Capacity is in H26:L31 (row 26 corresponds to row 35, row 27 to row 36, etc.)

So for H35: `=(H12-H19)/H26*100`
For H36: `=(H13-H20)/H27*100`
etc.

Verify the row mapping by checking hospital names in column D/E/F for rows 12-17 vs 19-24 vs 26-31 vs 35-40.

## 4. Populate H42:L47 with summary statistics

For each column (H through L):
- H42: `=MIN(H35:H40)` (minimum)
- H43: `=MAX(H35:H40)` (maximum)
- H44: `=MEDIAN(H35:H40)` (median)
- H45: `=AVERAGE(H35:H40)` (simple mean)
- H46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- H47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

Check the labels in column D/E/F for rows 42-47 to confirm the correct order of these statistics. Adjust row assignments if the labels differ from the order above.

## 5. Populate H50:L50 with weighted mean

For each column:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean of Net Patient Flow using Effective Bed Capacity as weights.

## 6. Save and verify
- Save the workbook (it should already be at `/root/output/result.xlsx`)
- Reopen it and verify that all formula cells are populated (not empty/None)
- Spot-check a few formulas by printing their string representation
- Confirm no new sheets were added
- Confirm the file exists at `/root/output/result.xlsx`

## IMPORTANT NOTES
- Use openpyxl for all Excel manipulation
- When writing formulas, prefix them with `=` and write them as strings (openpyxl stores formula strings)
- Do NOT use `data_only=True` when opening for writing
- Preserve all existing formatting — do not clear cells or overwrite non-yellow cells
- Before writing any formula, re-read the current cell to make sure you're not overwriting existing content that shouldn't be changed
- If the Data sheet layout is unexpected (e.g., transposed, or uses different column positions), adapt the INDEX/MATCH formula accordingly rather than forcing a wrong pattern
- Pay very careful attention to whether Data rows 21:38 have a header row (row 20?) with years, and where the series codes column is

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