# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`:

1. **Inspect the workbook** (`/root/data/workbook.xlsx`):
   - Open with openpyxl (data_only=False) and inspect both sheets.
   - On sheet `Data`: confirm the layout — row 22 as header, column A as series code key, columns B:G (or similar) as value columns, rows 21:38 as the data range.
   - On sheet `Task`: confirm column D holds series codes, row 10 holds years, and identify the exact cell references for the three blocks (H12:L17, H19:L24, H26:L31), the Net SLA buffer block (H35:L40), statistics block (H42:L47), and weighted mean row (H50:L50).
   - Print out the contents of row 10 (columns H through L) to get the year headers.
   - Print out column D for rows 12-17, 19-24, 26-31 to get the series codes.
   - Print out the Data sheet structure: row 21 and 22 contents, column A for rows 21-38, and a few sample data cells.

2. **Populate lookup formulas in H12:L17, H19:L24, H26:L31**:
   - For each cell, use an INDEX/MATCH formula pattern:
     `=INDEX(Data!$B$22:$G$38, MATCH($D{row}, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$G$21, 0))`
   - Adjust the ranges based on what you found in step 1. The key is:
     - Row match: match the series code in column D against the series code column on Data sheet.
     - Column match: match the year in row 10 against the year header row on Data sheet.
   - Verify the ranges are correct by checking a couple of expected intersections manually.

3. **Calculate Net SLA buffer in H35:L40**:
   - For each cell in H35:L40, the formula is:
     `=(H12-H19)/H26*100` (adjusted for the correct row offsets)
   - Specifically, for row 35 col H: `=(H12-H19)/H26*100`, row 36 col H: `=(H13-H20)/H27*100`, etc.
   - The pattern: row 35+i uses (row 12+i - row 19+i) / row 26+i * 100 for i=0..5.

4. **Calculate summary statistics in H42:L47**:
   - Row 42 (minimum): `=MIN(H35:H40)` for each column H through L.
   - Row 43 (maximum): `=MAX(H35:H40)` for each column.
   - Row 44 (median): `=MEDIAN(H35:H40)` for each column.
   - Row 45 (mean): `=AVERAGE(H35:H40)` for each column.
   - Row 46 (25th percentile): `=PERCENTILE.INC(H35:H40,0.25)` for each column.
   - Row 47 (75th percentile): `=PERCENTILE.INC(H35:H40,0.75)` for each column.
   
   **CRITICAL**: The cross-task feedback shows that `#NAME?` errors occurred for percentile rows in a similar task. This was likely caused by using `PERCENTILE.INC` which openpyxl may not handle, OR the Excel evaluation engine didn't recognize it. To be safe:
   - First try `PERCENTILE.INC` (the standard modern Excel function).
   - But also verify: check the labels in column D or G for rows 42-47 to confirm which row is which statistic. The order might be different from what's assumed above — read the actual labels before assigning formulas.
   - If the verifier evaluates with an engine that doesn't support `PERCENTILE.INC`, use `PERCENTILE` instead. Since the previous successful run used string-based formula assignment and got reward 1.0, replicate whatever approach worked. Check if `PERCENTILE.INC` or `PERCENTILE` is appropriate by looking at what the previous successful execution likely used.

5. **Calculate weighted mean in H50:L50**:
   - For each column H through L:
     `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`
   - This uses the Net SLA buffer percentages as values and Covered Request Capacity as weights.

6. **Save**: Copy the workbook to `/root/output/result.xlsx` (create `/root/output/` if needed). Do NOT add sheets, macros, VBA, or helper tabs.

7. **Validate**: Reopen the saved file with openpyxl (data_only=False) and spot-check that formulas are present in the expected cells (H12, H35, H42, H46, H50). Print a few formula strings to confirm correctness.

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