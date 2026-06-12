# Task Instruction

Execute the following multi-phase plan to populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx.

## Phase 0: Setup
```
import os
os.makedirs('/root/output', exist_ok=True)
```

## Phase 1: Inspect the workbook structure
Using openpyxl, open `/root/data/workbook.xlsx` with data_only=False. Inspect:
1. **Task sheet**: Print cells A10:L10 (year headers), D12:D17 (series codes block 1), D19:D24 (series codes block 2), D26:D31 (series codes block 3), A12:G17 (labels block 1), A35:G40 (labels block 3 / net patient flow), A42:A47 (stat labels), H50:L50 area, and any existing formulas or values already present.
2. **Data sheet**: Print rows 1-5 to see headers. Print column A (or B) for rows 21-38 to see series codes. Print the row that contains year headers. Identify the exact layout: which row has years, which column has series codes, and where the data matrix starts.

Print all findings clearly before proceeding.

## Phase 2: Construct and write lookup formulas
Based on Phase 1 inspection, write INDEX/MATCH formulas into H12:L17, H19:L24, and H26:L31.

The pattern for each cell should be:
```
=INDEX(Data!<data_range>, MATCH($D<row>, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Critical details:
- `$D<row>`: Lock column D so it stays fixed when copying across columns. The row number changes per row.
- `H$10`: Lock row 10 so it stays fixed when copying down rows. The column letter changes per column (H, I, J, K, L).
- `Data!<data_range>`: The rectangular block of numeric data on the Data sheet (rows 21:38, excluding the label column and year header row).
- `Data!<series_code_column>`: The column on Data sheet containing the series codes, spanning rows 21:38.
- `Data!<year_header_row>`: The row on Data sheet containing the year values, spanning the data columns.

Make sure the ranges are correct by cross-referencing with the Phase 1 inspection output. Use absolute references where needed (e.g., Data!$A$21:$A$38 for the series code column).

Write formulas cell by cell or in loops. For each block (H12:L17, H19:L24, H26:L31), iterate over 6 rows and 5 columns.

## Phase 3: Write Net Patient Flow formulas in H35:L40
For each of the 6 hospitals (rows 35-40) and 5 years (columns H-L):
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to Patient Admissions, H19 to Patient Discharges, and H26 to Effective Bed Capacity. Adjust row references for each hospital row:
- Row 35 uses rows 12, 19, 26
- Row 36 uses rows 13, 20, 27
- Row 37 uses rows 14, 21, 28
- Row 38 uses rows 15, 22, 29
- Row 39 uses rows 16, 23, 30
- Row 40 uses rows 17, 24, 31

## Phase 4: Write summary statistics in H42:L47
For each column H through L:
- Row 42: MIN of H35:H40 → `=MIN(H35:H40)`
- Row 43: MAX of H35:H40 → `=MAX(H35:H40)`
- Row 44: MEDIAN of H35:H40 → `=MEDIAN(H35:H40)`
- Row 45: AVERAGE of H35:H40 → `=AVERAGE(H35:H40)`
- Row 46: 25th percentile → `=PERCENTILE(H35:H40, 0.25)`
- Row 47: 75th percentile → `=PERCENTILE(H35:H40, 0.75)`

IMPORTANT: Check the actual labels in A42:A47 during Phase 1 to confirm the correct order of statistics. Adjust the row assignments if the labels differ from the assumed order above.

## Phase 5: Write weighted mean in H50:L50
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of net patient flow percentages using Effective Bed Capacity as weights.

## Phase 6: Save and verify
1. Save the workbook to `/root/output/result.xlsx` using `wb.save('/root/output/result.xlsx')`.
2. Reopen the saved file with data_only=False and print a sample of cells (e.g., H12, H19, H26, H35, H42, H50) to confirm formulas are present (not None).
3. Also reopen with data_only=True to check if any values resolve (they may show None in openpyxl without a calc engine, which is expected — the key is that formulas are present when data_only=False).

## Important Notes
- Use `openpyxl` throughout. Do NOT use data_only=True when opening for writing.
- Do NOT create new sheets, delete sheets, or add macros.
- Do NOT change any existing formatting, values, or structure.
- When writing formulas as strings, make sure they start with '='.
- Use the exact column letters and row numbers discovered in Phase 1.
- If the Data sheet year headers are in a different row than expected, or series codes are in a different column, adjust all formulas accordingly.
- Pay special attention to whether years on the Task sheet (row 10) are integers or strings, and whether they match the format on the Data sheet. If they differ in type, the MATCH may fail — consider wrapping in a type conversion if needed.

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