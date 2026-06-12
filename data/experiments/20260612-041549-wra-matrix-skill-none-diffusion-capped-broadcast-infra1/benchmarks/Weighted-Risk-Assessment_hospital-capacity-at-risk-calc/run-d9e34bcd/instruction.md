# Task Instruction

Execute the following steps exactly:

1. **Inspect the workbook** – Open `/root/data/workbook.xlsx` with openpyxl and inspect:
   - Sheet `Task`: read the layout of rows 10-50, especially column D (series codes), row 10 (years in H-L), the yellow target ranges, and any existing content/formatting.
   - Sheet `Data`: read rows 21-38 to understand the data layout (which row holds which series, which columns hold which years).
   Print enough to understand the exact structure before writing any formulas.

2. **Write lookup formulas in H12:L17, H19:L24, H26:L31** – For each cell in these three 6×5 blocks, write a formula that looks up the value from `Data!$A$21:$S$38` (adjust the range after inspection) using the series code in column D of the current row and the year in row 10. Use the pattern `INDEX(Data!<data_range>, MATCH($D12, Data!<series_column>, 0), MATCH(H$10, Data!<year_row>, 0))` — adjust references after inspecting the actual layout. Make sure:
   - The series-code reference locks the column (`$D12`).
   - The year reference locks the row (`H$10`).
   - References to the Data sheet are correct (sheet name, row/column ranges).

3. **Write Net capacity headroom formulas in H35:L40** – For each of the 6 rows and 5 year-columns, write:
   `=(H12 - H19) / H26 * 100`
   adjusting row references so that row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc. (i.e., the i-th cluster's Available Care Slots minus Occupied Care Slots, divided by Staffed Bed Capacity, times 100).

4. **Write summary statistics in H42:L47** – For each column H through L:
   - Row 42: `=MIN(H35:H40)`
   - Row 43: `=MAX(H35:H40)`
   - Row 44: `=MEDIAN(H35:H40)`
   - Row 45: `=AVERAGE(H35:H40)`
   - Row 46: `=PERCENTILE(H35:H40,0.25)`  ← Use `PERCENTILE`, **not** `PERCENTILE.INC`
   - Row 47: `=PERCENTILE(H35:H40,0.75)`  ← Use `PERCENTILE`, **not** `PERCENTILE.INC`

   **Critical**: The previous run failed because the evaluator did not recognize the function name. Use exactly `PERCENTILE` (no `.INC` or `.EXC` suffix).

5. **Write weighted mean in H50:L50** – For each column H through L:
   `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

6. **Preserve formatting** – Do not change any existing formatting, sheet names, or structure. Do not add sheets, macros, VBA, external links, or helper tabs.

7. **Save** – Save the workbook to `/root/output/result.xlsx` (create the `/root/output/` directory if it doesn't exist).

8. **Validate** – Re-open `/root/output/result.xlsx` with openpyxl and spot-check:
   - That cells in H12, L17, H19, L24, H26, L31 contain formula strings (start with '=').
   - That cells in H35, L40 contain formula strings.
   - That cells in H42:L47 contain formula strings, and specifically that H46 and H47 use `PERCENTILE` (not `PERCENTILE.INC`).
   - That H50:L50 contain `SUMPRODUCT` formulas.
   Print the formula strings for these cells to confirm correctness.

9. **Run the verifier** if a test script exists:
   ```
   cd /root && python -m pytest test_output.py -v 2>&1 | head -80
   ```

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