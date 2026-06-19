# Task Instruction

Execute the following steps carefully to produce /root/output/result.xlsx.

## Phase 0 – Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## Phase 1 – Inspect the workbook structure
Open /root/data/workbook.xlsx with openpyxl (data_only=False) and print:
1. Sheet names.
2. From sheet 'Task':
   - Row 10 (columns A–L) to see the year headers.
   - Column D rows 12–17 to see series codes for block 1.
   - Column D rows 19–24 to see series codes for block 2.
   - Column D rows 26–31 to see series codes for block 3.
   - Row 35–40 column A–D to understand the Net budget buffer block layout.
   - Row 42–47 column A–G to understand the summary stats labels.
   - Row 50 column A–G to understand the weighted mean row.
   - Cells H12:L12 to check if they are already populated or empty.
3. From sheet 'Data':
   - Rows 21–38, columns A through at least column P (print all values) so you can see the full data matrix layout: where series codes are, where year headers are, and the numeric data range.

Print everything clearly with row/column indices. Do NOT write any formulas yet.

## Phase 2 – Determine exact ranges
From the Phase 1 output, identify:
- The column in 'Data' that contains the series codes (likely column A or B).
- The row in 'Data' that contains the year headers.
- The rectangular data range on 'Data' that holds the numeric values.
- The first and last columns of the year range on 'Task' (H through L, i.e., columns 8–12).
- Confirm that H10:L10 on 'Task' contains years matching the 'Data' sheet header row.

## Phase 3 – Write lookup formulas in H12:L17, H19:L24, H26:L31
Using INDEX/MATCH/MATCH, write formulas into each cell. The pattern for cell (r, c) should be:
```
=INDEX(Data!<data_range>, MATCH($D{row}, Data!<series_code_column_range>, 0), MATCH({TaskSheetYearRef}, Data!<year_header_row_range>, 0))
```
where:
- `$D{row}` is an absolute column reference to the series code in column D of the current row on 'Task'.
- `{TaskSheetYearRef}` is a reference like `H$10` (absolute row, relative column) pointing to the year in row 10.
- `Data!<data_range>` is the rectangular numeric block on 'Data'.
- `Data!<series_code_column_range>` is the column of series codes on 'Data'.
- `Data!<year_header_row_range>` is the row of year headers on 'Data'.

Use absolute references ($) appropriately so the formula can be applied across the 5-column × 6-row blocks.

After writing, re-read a sample cell (e.g., H12) to confirm the formula string is stored.

## Phase 4 – Write Net budget buffer formulas in H35:L40
The formula for each cell is:
```
=(H{committed_row} - H{operating_row}) / H{approved_row} * 100
```
where for department i (i=0..5):
- committed_row = 12+i (H12:L17 block)
- operating_row = 19+i (H19:L24 block)
- approved_row = 26+i (H26:L31 block)
- result_row = 35+i

So H35 = (H12 - H19) / H26 * 100, H36 = (H13 - H20) / H27 * 100, etc.
Apply across columns H–L.

## Phase 5 – Summary statistics in H42:L47
For each column (H through L), write:
- Row 42: =MIN(H35:H40)
- Row 43: =MAX(H35:H40)
- Row 44: =MEDIAN(H35:H40)
- Row 45: =AVERAGE(H35:H40)
- Row 46: =PERCENTILE(H35:H40, 0.25)
- Row 47: =PERCENTILE(H35:H40, 0.75)

Verify the labels in column A/B/C/D of rows 42–47 match this ordering (min, max, median, mean, 25th, 75th). If the order differs, adjust accordingly.

## Phase 6 – Weighted mean in H50:L50
For each column c in H–L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
(adjusting column letter for each). This computes the weighted mean of Net budget buffer percentages weighted by Approved Budget Base.

## Phase 7 – Save and validate
1. Save the workbook to /root/output/result.xlsx.
2. Re-open the saved file and print:
   - Formulas in H12, L17, H19, L24, H26, L31 (spot-check lookups)
   - Formulas in H35, L40 (spot-check derived)
   - Formulas in H42, H47 (spot-check stats)
   - Formula in H50 (spot-check weighted mean)
3. Confirm no new sheets were added.
4. Confirm formatting was not altered (no explicit style changes were made).

IMPORTANT: Use openpyxl throughout. Do NOT use data_only=True when writing formulas. When writing formulas as strings, make sure they start with '='. Reference the 'Data' sheet as `Data!` in formulas. Use the exact range addresses determined in Phase 2.

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