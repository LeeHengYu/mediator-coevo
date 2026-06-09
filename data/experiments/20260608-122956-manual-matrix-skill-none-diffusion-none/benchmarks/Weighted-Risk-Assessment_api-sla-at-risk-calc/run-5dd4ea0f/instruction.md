# Task Instruction

Execute the following phases to build `/root/output/result.xlsx` from `/root/data/workbook.xlsx`.

## Phase 0 – Inspection
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False).
3. Print sheet names.
4. On sheet **Task**:
   - Print rows 10-50, columns A-L (values and any fill colours) so you can see:
     a. The year headers in row 10 (H10:L10) – note whether they are integers or strings.
     b. The series codes in D12:D17, D19:D24, D26:D31 – note exact strings (strip nothing yet).
     c. The labels in column A or B for rows 12-17, 19-24, 26-31 to understand which block is Latency Budget Preserved, Latency Budget Consumed, and Covered Request Capacity.
     d. Rows 35-50 to see existing labels for Net SLA buffer, statistics, and Platform SLA Coalition.
5. On sheet **Data**:
   - Print rows 21-38, all used columns, to see:
     a. Which column holds the series codes (likely column A or B).
     b. Which row holds the year headers.
     c. The exact data layout dimensions.
   - Also check the row just above row 21 (row 20) for year headers if the data header row is separate from the data rows.
6. Record:
   - `YEAR_ROW` on Data sheet (the row containing year values matching H10:L10).
   - `CODE_COL` on Data sheet (the column letter containing series codes).
   - `DATA_START_ROW`, `DATA_END_ROW` for the numeric data rows.
   - `DATA_START_COL`, `DATA_END_COL` for the numeric data columns.
   - Confirm that the series codes in Task D12:D17 etc. appear verbatim in the Data sheet's code column. If there are mismatches (whitespace, case), note them.
   - Identify which Task block (rows 12-17, 19-24, 26-31) corresponds to Latency Budget Preserved, Latency Budget Consumed, and Covered Request Capacity by reading labels.

## Phase 1 – Lookup Formulas (H12:L17, H19:L24, H26:L31)
Using the coordinates from Phase 0, write INDEX/MATCH formulas into every cell of the three 6×5 blocks.

Template (adjust references from Phase 0 findings):
```
=INDEX(Data!$<DATA_START_COL>$<DATA_START_ROW>:$<DATA_END_COL>$<DATA_END_ROW>,
       MATCH($D12,Data!$<CODE_COL>$<DATA_START_ROW>:$<CODE_COL>$<DATA_END_ROW>,0),
       MATCH(H$10,Data!$<DATA_START_COL>$<YEAR_ROW>:$<DATA_END_COL>$<YEAR_ROW>,0))
```

- `$D12` uses absolute column, relative row so it shifts down within each block.
- `H$10` uses relative column, absolute row so it shifts across columns.
- Apply this to all 90 cells (3 blocks × 6 rows × 5 columns).

**Critical**: If Phase 0 reveals that year headers are stored as integers in one sheet and strings/floats in another, wrap the MATCH search value with `VALUE()` or `TEXT()` as needed, or convert the Task row-10 cells to match. Prefer keeping formulas simple; if types already match, no wrapping is needed.

## Phase 2 – Net SLA Buffer (H35:L40)
For each of the 6 services (rows 35-40) and 5 years (columns H-L):
```
=(H12-H19)/H26*100
```
where:
- Row 12 maps to row 35 (first service), row 13→36, …, row 17→40
- Row 19 maps to Consumed for the same service (row 19→35, 20→36, …, 24→40)
- Row 26 maps to Capacity for the same service (row 26→35, 27→36, …, 31→40)

**Verify the block-to-meaning mapping from Phase 0 before writing these.** The formula is:
`(Preserved - Consumed) / Capacity * 100`
If the block order is different from rows 12/19/26, adjust accordingly.

## Phase 3 – Statistics (H42:L47)
For each year column (H through L), in the designated rows:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

**Check the labels in column A/B for rows 42-47 during Phase 0 to confirm which row is which statistic.** Adjust row assignments if the order differs.

## Phase 4 – Weighted Mean (H50:L50)
For each year column:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This uses Net SLA Buffer values as the values and Covered Request Capacity as weights.

## Phase 5 – Save and Validate
1. Save to `/root/output/result.xlsx` (keep formatting, do not add sheets/macros).
2. Reopen the saved file with data_only=False and print all formula cells to confirm they are formulas (not None, not hardcoded).
3. Reopen with data_only=True (or use xlcalc/formulas evaluation if available) and spot-check a few lookup cells and the weighted mean to see if they resolve to reasonable numbers. If data_only=True returns None (common without Excel engine), that's acceptable – the formula presence check is sufficient.
4. Confirm the file has exactly the original sheets (Task, Data) and no extras.

## Constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify formatting, existing values, or the Data sheet.
- Use openpyxl for all operations.
- If any formula produces an error during spot-check, debug by comparing series codes and year values between sheets.

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