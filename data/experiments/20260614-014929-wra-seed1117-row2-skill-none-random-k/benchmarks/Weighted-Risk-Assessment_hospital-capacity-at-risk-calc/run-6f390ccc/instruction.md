# Task Instruction

Implement the hospital-capacity-at-risk-calc task by following these steps:

1. **Inspect the workbook structure first.**
   - Open `/root/data/workbook.xlsx` using openpyxl (data_only=False).
   - Print the sheet names to confirm `Task` and `Data` exist.
   - Print the `Task` sheet contents in the range D10:L50 to understand the layout: column headers in row 10 (years), series codes in column D, labels, and which cells are yellow/empty.
   - Print the `Data` sheet rows 21:38 to understand the source data layout (column headers, series codes, years, values).
   - Identify the exact column structure of the Data sheet (which column has series codes, which columns have year values, etc.).

2. **Populate H12:L17, H19:L24, H26:L31 with INDEX/MATCH formulas.**
   - Each formula should look up the value from the `Data` sheet rows 21:38 using:
     - The series code from column D of the current row on `Task` sheet.
     - The year from row 10 of the current column on `Task` sheet.
   - Use the pattern: `=INDEX(Data!<value_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))`
   - Determine the exact cell references by inspecting the Data sheet layout. The value range, series code column, and year header row must match the actual Data sheet structure.
   - Apply formulas to all 6 rows × 5 columns in each of the three blocks (H12:L17, H19:L24, H26:L31) using appropriate relative/absolute references ($D for the series code column, $10 for the year row).

3. **Calculate Net capacity headroom in H35:L40.**
   - Formula: `(Available Care Slots - Occupied Care Slots) / Staffed Bed Capacity * 100`
   - Identify which block corresponds to each metric:
     - Look at the labels/series codes to determine which block (rows 12-17, 19-24, 26-31) is Available Care Slots, Occupied Care Slots, and Staffed Bed Capacity.
   - For each cell in H35:L40, write a formula like: `=(H12-H19)/H26*100` (adjust row references based on actual block assignments).

4. **Calculate statistics in H42:L47.**
   - Row 42: MIN of H35:H40 (column-wise) → `=MIN(H35:H40)`
   - Row 43: MAX → `=MAX(H35:H40)`
   - Row 44: MEDIAN → `=MEDIAN(H35:H40)`
   - Row 45: AVERAGE → `=AVERAGE(H35:H40)`
   - Row 46: 25th percentile → `=PERCENTILE(H35:H40, 0.25)` — **IMPORTANT: use PERCENTILE, not PERCENTILE.INC or PERCENTILE.EXC, as openpyxl may not support the dotted versions and they can cause #NAME? errors in some Excel engines.**
   - Row 47: 75th percentile → `=PERCENTILE(H35:H40, 0.75)`
   - **Check the labels in column D or nearby columns for rows 42-47 to confirm the exact order of statistics. Do NOT assume the order; read it from the sheet.**
   - **Critical: Verify that the function names you use are supported. The failed cloud-reliability task got #NAME? errors from using unsupported function names. Stick to PERCENTILE (not PERCENTILE.INC).**

5. **Calculate weighted mean in H50:L50.**
   - Use SUMPRODUCT with the Step 2 percentages (H35:H40) as values and Staffed Bed Capacity (H26:H31) as weights.
   - Formula: `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

6. **Save the workbook.**
   - Save to `/root/output/result.xlsx`.
   - Ensure `/root/output/` directory exists (create if needed).

7. **Validation.**
   - Re-open the saved file and print a sample of cells (e.g., H12, H35, H42, H46, H50) to confirm formulas are present and correctly structured.
   - If there's a test script (check for test_output.py or similar in the task directory), run it to verify.
   - Look for the test at `/root/test_output.py` or in the task directory and run `pytest` if found.

**Key warnings:**
- Do NOT use dotted function names like PERCENTILE.INC — use PERCENTILE instead.
- Do NOT add sheets, macros, VBA, or external links.
- Do NOT change existing formatting.
- Read the actual sheet layout before writing any formulas. The block-to-metric mapping must come from inspecting the sheet, not from assumptions.

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