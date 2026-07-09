# Task Instruction

Execute the following steps exactly:

1. **Inspect the workbook structure.**
   - Open `/root/data/workbook.xlsx` using openpyxl.
   - On sheet `Task`: read the series codes in column D for rows 12-17, 19-24, 26-31. Read the year headers in row 10 for columns H-L. Note the exact text/values so formulas reference them correctly.
   - On sheet `Data`: inspect rows 21-38 to understand the layout (which row has headers, which column has series codes, which columns have year data). Identify the column that contains the series codes and the row that contains year headers.

2. **Populate lookup formulas in H12:L17, H19:L24, H26:L31.**
   For each cell in these ranges, write an INDEX-MATCH formula that:
   - Uses the series code from column D of the current row.
   - Uses the year from row 10 of the current column.
   - Looks up in the Data sheet rows 21:38.
   - Pattern: `=INDEX(Data!$B$21:$S$38, MATCH($D12,Data!$A$21:$A$38,0), MATCH(H$10,Data!$B$20:$S$20,0))`
   - **IMPORTANT**: Before writing formulas, verify the exact column/row layout on Data sheet. Adjust the ranges accordingly:
     - Identify which column on Data holds the series codes (likely column A or B).
     - Identify which row on Data holds the year headers (likely row 20 or 21).
     - Identify the data range that spans the value cells.
   - Use absolute references for the lookup arrays and mixed references ($D12 for row-relative series code, H$10 for column-relative year).

3. **Calculate Net capacity headroom in H35:L40.**
   The formula for each cell is:
   `=(H12 - H19) / H26 * 100`
   where row 12-17 = Available Care Slots, row 19-24 = Occupied Care Slots, row 26-31 = Staffed Bed Capacity. Map each of the 6 clusters:
   - H35 = `=(H12-H19)/H26*100`, H36 = `=(H13-H20)/H27*100`, ... H40 = `=(H17-H24)/H31*100`
   - Same pattern across columns I through L.

4. **Calculate summary statistics in H42:L47.**
   For each column (H through L):
   - Row 42 (Minimum): `=MIN(H35:H40)`
   - Row 43 (Maximum): `=MAX(H35:H40)`
   - Row 44 (Median): `=MEDIAN(H35:H40)`
   - Row 45 (Mean): `=AVERAGE(H35:H40)`
   - Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
   - Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`
   - **CRITICAL**: Use plain `PERCENTILE`, NOT `PERCENTILE.INC` or `PERCENTILE.EXC`. The dotted variants cause #NAME? errors.

5. **Calculate weighted mean in H50:L50.**
   For each column (H through L):
   `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

6. **Verify the row-to-block mapping.**
   Before writing formulas, confirm by reading the Task sheet labels:
   - Rows 12-17 block label (should be Available Care Slots or similar)
   - Rows 19-24 block label (should be Occupied Care Slots or similar)
   - Rows 26-31 block label (should be Staffed Bed Capacity or similar)
   - Rows 35-40 block label (Net capacity headroom)
   - Rows 42-47 labels (min, max, median, mean, 25th, 75th — verify exact order)
   - Row 50 label (Regional Care Grid weighted mean)
   If the order of statistics rows differs from what I assumed, adjust accordingly.

7. **Save the workbook.**
   - Do NOT change any formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.
   - Save to `/root/output/result.xlsx` (create the output directory if needed).

8. **Validation.**
   - Reopen the saved file and spot-check a few formula cells to confirm they contain formula strings (not raw values).
   - Confirm the file has exactly the original sheets (Task and Data, no extras).
   - Confirm no cells outside the specified ranges were modified.

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