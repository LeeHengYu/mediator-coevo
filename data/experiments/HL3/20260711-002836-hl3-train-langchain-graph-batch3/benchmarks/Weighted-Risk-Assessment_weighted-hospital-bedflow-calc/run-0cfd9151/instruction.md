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
- On sheet `Task`: read row 10 (years in H10:L10), column D rows 12-31 (series codes), rows 26-31 col D (bed capacity series codes), row 35-40 labels, row 42-47 labels, row 50 label
- On sheet `Data`: inspect rows 21-38 to understand the data layout (what's in each column/row, where series codes are, where years are as headers)
- Check which cells are yellow (H12:L17, H19:L24, H26:L31) to confirm they are empty and need formulas
- Print all findings clearly before proceeding

## 2. Understand the Data sheet layout
Determine:
- Are series codes in a column (e.g., column A) on Data rows 21-38?
- Are years in a header row above the data?
- What is the exact range of the data table on the Data sheet?
This determines which lookup pattern to use. Print the first few cells of each relevant row/column.

## 3. Populate H12:L17, H19:L24, H26:L31 with lookup formulas (Step 1)
Using openpyxl, write Excel formulas into each cell. For each cell at position (row, col) where col is H-L (columns 8-12) and row is in the specified ranges:
- The series code is in column D of that same row on sheet Task
- The year is in row 10 of that same column on sheet Task
- The data source is on sheet Data rows 21:38

Use INDEX/MATCH pattern. The exact formula depends on the Data sheet layout discovered in step 2. A typical pattern would be:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```
Adjust the ranges based on what you found. Make sure:
- The row reference for the series code uses `$D` (absolute column) so it doesn't shift when copied across columns
- The column reference for the year uses `$10` (absolute row) so it doesn't shift when copied down rows
- All three blocks (H12:L17, H19:L24, H26:L31) use the same formula pattern since they all look up from the same Data source

## 4. Populate H35:L40 with Net Patient Flow formulas (Step 2)
Net patient flow = (Patient Admissions - Patient Discharges) / Effective Bed Capacity * 100

Based on the sheet layout:
- Patient Admissions should be in H12:L17
- Patient Discharges should be in H19:L24
- Effective Bed Capacity should be in H26:L31

For each cell in H35:L40, the formula should reference the corresponding cells. For example, H35:
```
=(H12-H19)/H26*100
```
Verify the row correspondence: row 35 corresponds to the first hospital, row 40 to the sixth. Make sure the hospital order matches between the admissions, discharges, capacity, and net flow blocks.

## 5. Populate H42:L47 with summary statistics (Step 2 continued)
For each column H through L:
- H42 (minimum): `=MIN(H35:H40)`
- H43 (maximum): `=MAX(H35:H40)`
- H44 (median): `=MEDIAN(H35:H40)`
- H45 (mean): `=AVERAGE(H35:H40)`
- H46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- H47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Check the row labels (rows 42-47) to confirm which statistic goes in which row. Adjust the mapping accordingly.

## 6. Populate H50:L50 with weighted mean using SUMPRODUCT (Step 3)
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of net patient flow percentages using effective bed capacity as weights.

## 7. Save and validate
- Save the workbook with openpyxl (do NOT use data_only mode; formulas must be preserved)
- Reopen and verify that formulas are written correctly in a sample of cells
- Confirm no new sheets were added
- Confirm the file is saved at `/root/output/result.xlsx`

## Critical Notes
- Use `openpyxl` for all Excel operations
- Write EXCEL FORMULAS as strings (starting with `=`), not computed Python values
- Do NOT use `data_only=True` when loading for writing
- Do NOT add any sheets, macros, or VBA
- Preserve all existing formatting (do not change fonts, colors, borders, etc.)
- When writing formulas, make sure cell references use the correct sheet name prefix `Data!` for cross-sheet references
- Double-check row/column mappings by printing the actual cell contents before writing formulas

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