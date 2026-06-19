# Task Instruction

Execute the following steps carefully to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl and inspect:
- Sheet names (confirm `Task` and `Data` exist)
- On sheet `Task`: read cells D12:D17, D19:D24, D26:D31 to get series codes; read H10:L10 to get years; read labels around rows 35-50 to understand the layout; check what's in H35:L40, H42:L47, H50:L50
- On sheet `Data`: read row 21 header and rows 21:38 to understand the data layout (column structure, where series codes and years appear)
- Print all of this so you understand the exact structure before writing any formulas

## 2. Understand the Data sheet layout
Determine:
- Whether Data rows 21:38 are organized with series codes in a column and years across columns (or vice versa)
- Which column contains the series codes on Data sheet
- Which row contains the year headers on Data sheet
- The exact cell references needed for lookups

## 3. Populate H12:L17, H19:L24, H26:L31 with lookup formulas
For each yellow cell in these ranges, write a formula that:
- Uses the series code from column D of that row on the Task sheet
- Uses the year from row 10 of that column on the Task sheet
- Looks up the value from Data!$21:$38
- Uses one of the allowed patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH

Choose the pattern that best fits the Data layout. For example, if Data has series codes in column A and years across the top row of the range:
- `INDEX(Data!<data_range>, MATCH(D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))` 

Make sure to:
- Use appropriate absolute/mixed references so formulas can span the range correctly (lock the series code column reference with $D, lock the year row with $10, lock the Data ranges appropriately)
- Write the formula as a string in openpyxl (do NOT use data_only mode)
- The formula must start with `=`

## 4. Populate H35:L40 with Net budget buffer formula
The formula is: `(Committed Funding - Operating Spend) / Approved Budget Base * 100`

Based on the Task sheet layout:
- H12:L17 appears to be one block (likely Committed Funding, Operating Spend, or Approved Budget Base)
- H19:L24 appears to be another block
- H26:L31 appears to be the third block

Read the labels in the Task sheet (likely in column B or nearby) to identify which block is which. Then for each cell in H35:L40, write a formula referencing the appropriate cells from the three blocks above. For example, if row 12 corresponds to the same department as row 35:
- `=(H12 - H19) / H26 * 100` (adjust based on actual block assignments)

Make sure the department ordering matches between the lookup blocks and the buffer block.

## 5. Populate H42:L47 with summary statistics
For each column H through L:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)  
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

IMPORTANT: Check the labels in column B/C/D for rows 42-47 to determine the exact order of these statistics. Match the formula to the label, not to my assumed order above.

## 6. Populate H50:L50 with weighted mean using SUMPRODUCT
For each column H through L:
- `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net budget buffer percentages (H35:H40) weighted by Approved Budget Base (H26:L31).

## 7. Save the workbook
- Save to `/root/output/result.xlsx`
- Do NOT change any formatting, do NOT add sheets, macros, VBA, external links, or helper tabs
- Use `wb.save('/root/output/result.xlsx')`

## 8. Verify
- Reopen the saved file and read back a sample of cells to confirm formulas are present (not None or empty)
- Check that sheets are still only `Task` and `Data`
- Confirm no extra sheets were added

## Critical Notes
- Use openpyxl WITHOUT data_only=True so formulas are preserved
- When writing formulas, they must be strings starting with `=`
- Do NOT evaluate formulas in Python; write them as Excel formula strings
- Read the actual cell labels carefully before assuming which block is Committed Funding, Operating Spend, or Approved Budget Base
- The Data sheet structure must be inspected first to write correct lookup formulas

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