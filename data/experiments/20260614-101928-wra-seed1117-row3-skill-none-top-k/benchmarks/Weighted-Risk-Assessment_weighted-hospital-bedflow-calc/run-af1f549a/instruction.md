# Task Instruction

Execute the following steps precisely:

## Step 0: Inspect the workbook
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Open `/root/output/result.xlsx` with openpyxl (with `data_only=False` so formulas are preserved).
3. Print the sheet names.
4. For sheet `Task`: print rows 1–55 (columns A–M) so we can see the layout, especially:
   - Column D (series codes for rows 12–31)
   - Row 10 (years in H10:L10)
   - The yellow cell ranges H12:L17, H19:L24, H26:L31
   - Rows 35–50
5. For sheet `Data`: print rows 1–40 (all used columns) to see the data layout, especially rows 21–38. Identify:
   - Where the series codes appear (likely in a column)
   - Where the years appear (likely in a row)
   - The data orientation

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

After inspecting the data layout, write formulas into each yellow cell. Each formula must:
- Reference the series code from column D of the same row (e.g., `$D12`)
- Reference the year from row 10 of the same column (e.g., `H$10`)
- Look up the value from sheet `Data` rows 21:38
- Use one of the allowed patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH

The recommended approach (adapt after seeing data layout):
- If data on `Data` sheet has series codes in a column and years in a header row, use:
  `=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))`
- Make sure the row reference to column D is absolute on column ($D) and the column reference to row 10 is absolute on row ($10) so the formula copies correctly across the grid.
- Write formulas for all 6 rows × 5 columns in each of the three blocks (90 cells total).

## Step 2: Net patient flow in H35:L40 and statistics in H42:L47

For H35:L40 (6 hospitals × 5 years):
- Formula: `=(H12 - H19) / H26 * 100` (adjusting row references per hospital)
  - H12:L17 = Patient Admissions
  - H19:L24 = Patient Discharges  
  - H26:L31 = Effective Bed Capacity
- So for cell H35: `=(H12-H19)/H26*100`, H36: `=(H13-H20)/H27*100`, etc.
- Verify the row mapping: row 35 corresponds to row 12/19/26, row 36 to 13/20/27, etc.

For H42:L47 (column-wise statistics over H35:L40):
- H42: `=MIN(H35:H40)` (minimum)
- H43: `=MAX(H35:H40)` (maximum)
- H44: `=MEDIAN(H35:H40)` (median)
- H45: `=AVERAGE(H35:H40)` (simple mean)
- H46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- H47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)
- Check the labels in column D/E/F/G for rows 42–47 to confirm the correct order of statistics. Adjust row assignments if the labels indicate a different order.

## Step 3: Weighted mean in H50:L50
- Formula: `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
- This uses SUMPRODUCT as required. The values are the net patient flow percentages and the weights are Effective Bed Capacity.
- Apply across H50:L50 for each year column.

## Step 4: Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the file and print the formula cells to confirm:
   - All 90 lookup cells contain formulas (not hardcoded values)
   - The net patient flow formulas reference the correct rows
   - Statistics formulas reference H35:H40 ranges
   - Weighted mean uses SUMPRODUCT
3. Verify no new sheets were added and existing formatting is intact.

**IMPORTANT**: After the initial inspection (Step 0), adapt all row/column references to match the actual layout before writing any formulas. Do not assume the layout—verify it first. Pay special attention to whether the statistics labels match the order I suggested (min, max, median, mean, 25th, 75th) and adjust accordingly.

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