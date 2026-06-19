# Task Instruction

Execute the following steps in order.

## Step 0 – Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and print:
1. Sheet names.
2. From sheet `Task`: rows 10-50, columns A-L (print cell values so we can see headers, series codes, year row, and which cells are empty/yellow).
3. From sheet `Data`: rows 1-40, all non-empty columns (print cell values to see the lookup table structure, header row, series codes, years, and data range).

Do NOT edit anything yet. Just print the inspection output and review it before proceeding.

## Step 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31
Based on the inspection output, construct INDEX/MATCH formulas for each cell in the three blocks. Each formula should use:
- The series code from column D of the current row on sheet `Task`.
- The year from row 10 on sheet `Task`.
- The data range on sheet `Data` (rows 21:38 as stated, but confirm exact columns from inspection).

Use the pattern: `=INDEX(Data!<data_range>, MATCH($D<row>, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))`

Adjust the exact references (sheet name, data range, series code column, year header row) based on what you found in Step 0. Use absolute references for the data range and series column, and mixed references ($D for column, H$10 pattern for year row) so formulas can be written once per block and adapted per cell.

Write formulas to all 90 cells (3 blocks × 6 rows × 5 columns).

## Step 2 – Net container flow (H35:L40) and summary statistics (H42:L47)
For each cell in H35:L40, write a formula:
`=(H12 - H19) / H26 * 100`
where H12, H19, H26 correspond to the same port (same relative row within each block) and same year column. Adjust row references for each of the 6 ports.

For H42:L47, write column-wise formulas over H35:L40:
- Row 42: `=MIN(H35:H40)` (or whichever row the label says is minimum – confirm from inspection)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

Check the actual row labels from Step 0 to assign the correct statistic to the correct row.

## Step 3 – Weighted mean for CPA (H50:L50)
For each column H-L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the Net container flow percentages as values and Terminal Throughput Capacity as weights.

## Step 4 – Save and verify
1. Save the workbook to `/root/output/result.xlsx` (create the output directory if needed).
2. Re-open the saved file with openpyxl (data_only=False) and print cells H12, H19, H26, H35, H42, H50 (all column H) to confirm they contain formula strings (not None).
3. Also spot-check a few cells in other columns (e.g., L17, L24, L31, L40, L47, L50) to confirm formulas are present.

Do NOT add sheets, macros, VBA, external links, or helper tabs. Do NOT change existing formatting. Only write formulas into the specified cells.

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