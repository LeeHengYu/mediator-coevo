# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0 – Inspect the workbook layout
1. Open /root/data/workbook.xlsx with openpyxl (data_only=False).
2. Print sheet names to confirm 'Task' and 'Data' exist.
3. On sheet 'Data': print rows 19–40 (columns A–Z or so) to identify:
   - Which row contains the year headers (likely row 21 or nearby).
   - Which column contains the series/code identifiers.
   - The data range rows 21:38.
4. On sheet 'Task': print rows 8–52, columns A–L to identify:
   - Row 10: year headers in columns H–L.
   - Column D: series codes for rows 12–17, 19–24, 26–31.
   - Row labels for rows 35–40 (the six services for Net SLA buffer).
   - Row labels for rows 42–47 (min, max, median, mean, 25th, 75th).
   - Row 50 label (Platform SLA Coalition weighted mean).
   - Which cells are currently empty / yellow (the target cells).

Print all of this before writing anything.

## 1 – Determine Data sheet anchor coordinates
From the inspection, note:
- DATA_HEADER_ROW: the row on 'Data' that contains year values matching Task!H10:L10.
- DATA_CODE_COL: the column on 'Data' that contains the series codes matching Task!D12:D31.
- DATA_RANGE: the row range 21:38 as stated in the task.

Build the INDEX/MATCH formula template:
```
=INDEX(Data!<data_range>, MATCH($D{row}, Data!$<code_col>$21:$<code_col>$38, 0), MATCH(H$10, Data!$<first_data_col>${header_row}:$<last_data_col>${header_row}, 0))
```
Adjust column letters and row numbers based on what you actually see.

## 2 – Write lookup formulas (Step 1)
For each target cell in H12:L17, H19:L24, H26:L31:
- Construct the INDEX/MATCH formula using the series code in column D of that row and the year in row 10.
- Use absolute references for the Data lookup ranges ($-signs) so they don't shift.
- The row reference to $D{row} should lock the column ($D) but keep the row relative to the current row.
- The column reference to H$10 (etc.) should keep the row locked at 10.
- Write the formula string into the cell.

## 3 – Write Net SLA buffer formulas (Step 2, rows 35–40)
From the inspection, identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to:
- "Latency Budget Preserved" (one block)
- "Latency Budget Consumed" (another block)
- "Covered Request Capacity" (the third block, H26:L31 per Step 3)

Read the block header labels (likely in rows 11, 18, 25 or nearby) to determine the mapping.

For each cell in H35:L40, write a formula:
```
=(Preserved_cell - Consumed_cell) / Capacity_cell * 100
```
where Preserved_cell, Consumed_cell, and Capacity_cell are the corresponding cells from the three blocks (same relative position: same column, same offset within each block).

The six services in rows 35–40 should correspond to the six rows within each block (rows 12–17, 19–24, 26–31). Verify the service names in column D match across blocks and rows 35–40.

## 4 – Write summary statistics formulas (Step 2, rows 42–47)
Read the labels in column D (or nearby) for rows 42–47 to determine which statistic goes where. Then for each column H–L:
- MIN: =MIN(H35:H40)
- MAX: =MAX(H35:H40)
- MEDIAN: =MEDIAN(H35:H40)
- AVERAGE: =AVERAGE(H35:H40)
- 25th percentile: =PERCENTILE(H35:H40, 0.25)
- 75th percentile: =PERCENTILE(H35:H40, 0.75)

Map each formula to the correct row based on the label.

## 5 – Write weighted mean formula (Step 3, row 50)
For each column H–L in row 50:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
(Using the Net SLA buffer percentages as values and Covered Request Capacity as weights.)

## 6 – Save and validate
1. Save to /root/output/result.xlsx (create /root/output/ if needed).
2. Re-open the saved file with openpyxl (data_only=False).
3. Print cells H12, L17, H19, L24, H26, L31 to confirm they contain formula strings (not None).
4. Print cells H35, L40, H42, L47, H50, L50 to confirm they contain formula strings.
5. Confirm no new sheets were added.

IMPORTANT:
- Do NOT use data_only=True when writing; open normally.
- Do NOT add sheets, macros, VBA, or external links.
- Preserve all existing formatting.
- Use openpyxl only; do not use xlsxwriter or pandas ExcelWriter in a way that recreates the workbook.

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