# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`:

1. **Inspect the workbook** – Open `/root/data/workbook.xlsx` with openpyxl. List sheet names, then inspect:
   - Sheet `Task`: read row 10 (years in H10:L10), column D for rows 12-17, 19-24, 26-31 (series codes), rows 35-40 labels, row 42-47 labels, row 50 label. Also note the campus names in column C or B for rows 12-17 and the weight/baseline block layout.
   - Sheet `Data`: read rows 21-38 to understand the data layout (which row holds which series code, which columns hold which years).
   Print enough to confirm the exact row/column structure before writing any formulas.

2. **Populate H12:L17, H19:L24, H26:L31 with INDEX/MATCH formulas** – For each cell in these three 6×5 blocks, write a formula of the form:
   ```
   =INDEX(Data!$B$21:$XX$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$XX$20,0))
   ```
   Adjust the exact column/row references after inspecting the Data sheet layout:
   - The MATCH for the series code should search the column in Data that contains series codes (likely column A or B).
   - The MATCH for the year should search the header row of the data range (likely row 20 or 21).
   - Use absolute references for the Data range and the lookup vectors; use mixed references ($D12 for the series code column, H$10 for the year row) so the formula can be applied across the block.
   
   Write formulas as strings (do NOT use `data_only`). Confirm the formula text in a sample cell after writing.

3. **Populate H35:L40 with Net Renewable Balance formulas** – For each of the 6 campus rows (35-40) and 5 year columns (H-L), write:
   ```
   =(H12-H19)/H26*100
   ```
   adjusting row references so that row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.

4. **Populate H42:L47 with statistical formulas** – For each year column (H-L):
   - H42: `=MIN(H35:H40)`
   - H43: `=MAX(H35:H40)`
   - H44: `=MEDIAN(H35:H40)`
   - H45: `=AVERAGE(H35:H40)`
   - H46: `=PERCENTILE(H35:H40,0.25)`
   - H47: `=PERCENTILE(H35:H40,0.75)`
   
   **Important**: Check the labels in rows 42-47 on the Task sheet to confirm which row is min, max, median, mean, 25th, 75th percentile and assign accordingly. Do NOT assume the order above; match the labels.

5. **Populate H50:L50 with SUMPRODUCT weighted mean** – For each year column:
   ```
   =SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
   ```

6. **Save** – Create `/root/output/` if needed. Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, add sheets, macros, or VBA.

7. **Validate** – Re-open `/root/output/result.xlsx` with openpyxl (without data_only) and print the formula strings in cells H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50 to confirm they are non-None formula strings starting with '='. If any cell is None, diagnose and fix before finishing.

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