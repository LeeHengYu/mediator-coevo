# Task Instruction

Execute the following steps to produce /root/output/result.xlsx.

## Phase 0 – Inspect the workbook layout
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False).
3. Print sheet names.
4. For sheet `Task`:
   - Print rows 10-50 for columns A-L (use cell.value). Pay special attention to:
     • Row 10 (years row) – which columns hold years and what are the year values.
     • Column D rows 12-31 – series codes for the three lookup blocks.
     • Rows 35-40 – labels for the six ports (Net container flow block).
     • Rows 42-47 – labels for min/max/median/mean/25th/75th.
     • Row 50 – CPA weighted mean row.
     • H26:L31 – Terminal Throughput Capacity block (used as weights in Step 3).
5. For sheet `Data`:
   - Print rows 19-40 for all used columns (at least A-Z). Identify:
     • Where the series codes appear (column or row).
     • Where the year headers appear (column or row).
     • The exact row/column ranges for the data table (rows 21:38 per the task).
   - Determine the orientation: are series codes in a column with years across a row, or vice versa?

**Do NOT write any formulas yet.** Just print and analyze.

## Phase 1 – Construct and write lookup formulas (Step 1)
Based on the layout discovered in Phase 0, write INDEX/MATCH formulas into the yellow cells H12:L17, H19:L24, H26:L31.

The formula pattern should be:
- `=INDEX(data_range, MATCH(series_code_ref, series_code_column, 0), MATCH(year_ref, year_row, 0))`
- Where `series_code_ref` = the cell in column D of the current row on `Task` sheet.
- Where `year_ref` = the cell in row 10 of the current column on `Task` sheet.
- `data_range`, `series_code_column`, and `year_row` must reference the `Data` sheet with absolute references.

**Critical:** Use the EXACT layout discovered in Phase 0. If the Data sheet has series codes in a row (not column), adjust to use the transposed INDEX/MATCH pattern or use HLOOKUP+MATCH instead. The cross-task artifacts warn that getting the orientation wrong causes None values.

Write formulas as strings using openpyxl (e.g., `ws['H12'] = '=INDEX(Data!$B$21:$F$38,MATCH(...),MATCH(...))'`). Adjust ranges to match actual data boundaries.

## Phase 2 – Net container flow formulas (Step 2, rows 35-40)
For each port (rows 35-40) and each year column (H-L):
- Identify which rows in the three lookup blocks correspond to:
  • Loaded Containers Inbound (block H12:L17)
  • Loaded Containers Outbound (block H19:L24)
  • Terminal Throughput Capacity (block H26:L31)
- The port order in rows 35-40 should match the port order in each block.
- Formula: `=(H12-H19)/H26*100` (adjusted for correct row offsets per port).

## Phase 3 – Statistical summary formulas (Step 2, rows 42-47)
For each year column H-L:
- Row 42 (min): `=MIN(H35:H40)`
- Row 43 (max): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Check the actual row labels to confirm which row is which statistic.

## Phase 4 – Weighted mean (Step 3, row 50)
For each year column H-L:
- `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

## Phase 5 – Save and verify
1. Save as `/root/output/result.xlsx`.
2. Re-open the saved file and print cells H12, H19, H26, H35, H42, H50 (column H) to confirm they contain formula strings (not None).
3. Optionally load with data_only=True to check if values are cached (they may be None since openpyxl can't evaluate, which is fine – the formulas just need to be present).

## Important constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting.
- Only write into the specified cell ranges.
- Use the `Task` and `Data` sheet names exactly as they appear in the workbook.

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