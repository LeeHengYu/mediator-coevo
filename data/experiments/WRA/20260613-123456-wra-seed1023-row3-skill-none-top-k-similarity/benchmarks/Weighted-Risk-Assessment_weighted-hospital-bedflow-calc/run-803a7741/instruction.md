# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

Follow these steps precisely in order.

### Step 0: Inspect the workbook structure

1. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False).
2. Print the sheet names.
3. On sheet `Task`:
   - Print rows 10-11 (to see the year headers in H10:L10).
   - Print rows 12-17 column D (to see the series codes for the first block).
   - Print rows 19-24 column D (series codes for the second block).
   - Print rows 26-31 column D (series codes for the third block — Effective Bed Capacity).
   - Print rows 35-40 column D (hospital names for Net patient flow).
   - Print row 42-47 column A-G (to see the stat labels: min, max, median, mean, 25th, 75th percentile).
   - Print row 50 columns A-G (to see the MHN weighted mean label).
4. On sheet `Data`:
   - Print row 1 (headers) to see the column layout.
   - Print rows 21-38, all columns, to see the actual data: identify which column holds the series codes and which columns/rows hold the year values.
   - Specifically note: Are years in a row (horizontal) or in a column (vertical)? Which column has the series code? What exact series code strings appear?
5. Print ALL of the above before writing any formulas. This is critical.

### Step 1: Write lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write INDEX/MATCH formulas. The general pattern should be:

```
=INDEX(Data!<data_range>, MATCH(<series_code_cell>, Data!<series_code_column>, 0), MATCH(<year_cell>, Data!<year_row>, 0))
```

Key considerations:
- The `<data_range>` must cover the full rectangular block of data on the Data sheet (rows 21:38, and the columns containing both the series codes and the numeric data).
- The `<series_code_cell>` reference is the cell in column D of the current row on the Task sheet (e.g., $D12 for row 12).
- The `<series_code_column>` is the column in Data that contains the series codes (likely column A or B of the Data sheet, within rows 21:38).
- The `<year_cell>` reference is the cell in row 10 of the Task sheet for the current column (e.g., H$10 for column H).
- The `<year_row>` is the row in Data that contains the year headers.
- Make sure the MATCH for years uses the correct row reference. If years are in row 1 of Data, use Data!<row1_range>. Adjust based on inspection.
- Use absolute references where appropriate ($D12 for series code column, H$10 for year row).

**IMPORTANT**: After inspection, verify the exact series code strings in column D of Task sheet match those in the Data sheet. If they don't match exactly (case, spacing, etc.), the MATCH will fail and return None. Print both sets of strings to compare.

**IMPORTANT**: If the Data sheet has years as numbers (e.g., 2019) and the Task sheet has them as numbers too, MATCH should work. But if one is text and the other numeric, there will be a mismatch. Check the types.

Write the formulas using openpyxl by setting `cell.value = '=INDEX(...)'` for each cell in the three blocks.

### Step 2: Net patient flow formulas in H35:L40

For each hospital (rows 35-40) and each year column (H-L):
```
= (Patient_Admissions - Patient_Discharges) / Effective_Bed_Capacity * 100
```

The Patient Admissions values are in H12:L17, Patient Discharges in H19:L24, and Effective Bed Capacity in H26:L31. So for cell H35:
```
= (H12 - H19) / H26 * 100
```
And similarly for the other cells, matching the row offsets (row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.).

### Step 2b: Statistics in H42:L47

For each column (H through L), calculate column-wise statistics over H35:L40:
- Row 42: `=MIN(H35:H40)` (or whichever row is minimum based on the label order from inspection)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

**CRITICAL**: Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC`. The `.INC`/`.EXC` variants cause `#NAME?` errors in some Excel engines and in openpyxl evaluation. Similarly use `AVERAGE` not `MEAN`. Check the labels from inspection to confirm the correct order of min/max/median/mean/25th/75th.

### Step 3: Weighted mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

This computes the weighted mean of Net patient flow percentages using Effective Bed Capacity as weights.

### Step 4: Save

1. Create `/root/output/` directory if it doesn't exist.
2. Save the workbook to `/root/output/result.xlsx`.
3. Do NOT change any existing formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.

### Step 5: Validate

1. Reopen `/root/output/result.xlsx` with openpyxl (data_only=False).
2. Print the formula strings in cells H12, L17, H19, L24, H26, L31, H35, H40, H42, H47, H50, L50.
3. Confirm they are formula strings (start with '='), not None or error values.
4. If any cell is None, debug immediately — re-inspect the workbook and fix.

### Step 6: Run the test

Look for a test file (e.g., `test_output.py` or similar) in the task directory and run it with pytest to verify correctness. If tests fail, read the error messages carefully and fix the formulas accordingly.

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