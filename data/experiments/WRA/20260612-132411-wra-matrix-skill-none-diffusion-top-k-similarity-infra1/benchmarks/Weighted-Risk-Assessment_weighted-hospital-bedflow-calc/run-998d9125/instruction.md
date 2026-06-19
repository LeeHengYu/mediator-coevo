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
- On sheet `Task`: read rows 10-50, columns D-L to understand the layout:
  - Row 10: should contain year headers in columns H-L
  - Column D rows 12-17, 19-24, 26-31: should contain series codes
  - Rows 35-40: hospital names for Net patient flow
  - Rows 42-47: labels for min, max, median, mean, 25th/75th percentile
  - Row 50: MHN weighted mean
- On sheet `Data`: read rows 21-38 to understand the lookup source structure (what's in each column, how series codes and years are arranged)

Print all of this information so you can construct correct formulas.

## 2. Understand the Data sheet layout
Determine:
- Whether Data rows 21-38 are arranged with series codes in a column and years across columns (suitable for VLOOKUP/INDEX-MATCH), or years in a row and series codes across (suitable for HLOOKUP)
- Which column contains the series codes and which row contains the year headers on the Data sheet
- The exact range boundaries

## 3. Write formulas using openpyxl
Use openpyxl to write formulas into the cells. Important rules:
- Use `openpyxl.load_workbook('/root/output/result.xlsx')` (do NOT use data_only=True)
- Write formula strings (starting with `=`) into cells
- Do NOT overwrite any cells outside the specified yellow ranges
- Do NOT change formatting, styles, or any other cell properties
- Reference the `Data` sheet as `Data` in formulas (e.g., `Data!$A$21:$Z$38`)

### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these ranges, write a formula that:
- Takes the series code from column D of the same row on the Task sheet
- Takes the year from row 10 of the same column on the Task sheet  
- Looks up the value from Data!rows 21:38

Use INDEX-MATCH-MATCH pattern (most reliable):
```
=INDEX(Data!<data_range>, MATCH($D{row}, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```
Adjust the exact ranges based on what you discovered in step 2. The `$D{row}` should use a mixed reference `$D12` (column absolute, row relative). The year reference should be `H$10` (row absolute, column relative) so the formula copies correctly across the range.

Make sure to lock references to the Data sheet ranges with absolute references (`$`) so they don't shift.

### Step 2: Net patient flow in H35:L40
Based on the task description, the three blocks are:
- H12:L17 = one metric (likely Patient Admissions)
- H19:L24 = another metric (likely Patient Discharges)  
- H26:L31 = another metric (likely Effective Bed Capacity)

Read the row labels in column D or nearby columns for rows 12-17, 19-24, 26-31 and rows 35-40 to confirm which block is which. The formula for each cell in H35:L40 is:
```
=(H{admissions_row} - H{discharges_row}) / H{capacity_row} * 100
```
where the admissions, discharges, and capacity rows correspond to the same hospital. Match hospitals by checking labels.

For H42:L47 (summary statistics), use:
- MIN: `=MIN(H35:H40)` (or `=MIN(H$35:H$40)`)
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- AVERAGE: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- 75th percentile: `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`

Check the labels in column D/E/F/G for rows 42-47 to determine which row gets which formula. Apply across columns H-L.

### Step 3: Weighted mean in H50:L50
For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net patient flow percentages weighted by Effective Bed Capacity.

## 4. Save and verify
- Save the workbook with `wb.save('/root/output/result.xlsx')`
- Reopen and verify that formulas are written in the correct cells
- Verify no extra sheets were added
- Verify the formula strings look correct by printing a sample

## Critical Notes
- You MUST inspect the actual workbook content before writing any formulas. The exact row/column references depend on the actual layout.
- Use `data_only=False` (the default) when loading to preserve and write formulas.
- Do not modify cell formatting/styles - only set `.value` on the target cells.
- If the Data sheet has a different structure than expected (e.g., transposed), adapt the lookup formula accordingly.
- Make sure all formula references are correct Excel syntax with proper sheet references like `Data!$A$21:$A$38`.
- Double-check that the 6 hospitals in rows 35-40 correspond to the same 6 hospitals in each of the three blocks (rows 12-17, 19-24, 26-31) in the same order.

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