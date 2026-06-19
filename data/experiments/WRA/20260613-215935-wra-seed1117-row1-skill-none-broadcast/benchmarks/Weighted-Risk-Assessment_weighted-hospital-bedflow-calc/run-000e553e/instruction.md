# Task Instruction

Reproduce the previously successful approach: open `/root/data/workbook.xlsx` with openpyxl, read the data from the `Data` sheet, compute all required values in Python, write them as numeric values into the `Task` sheet, and save to `/root/output/result.xlsx`.

Detailed steps:

1. **Setup**: `mkdir -p /root/output` and open `/root/data/workbook.xlsx` with `openpyxl` (NOT `data_only=True` — we want to preserve existing formatting).

2. **Understand the layout on sheet `Task`**:
   - Row 10 contains year headers in columns H through L (columns 8–12).
   - Column D (column 4) contains series codes for each row.
   - Rows 12–17: first data block (6 hospitals, one metric)
   - Rows 19–24: second data block (6 hospitals, another metric)
   - Rows 26–31: third data block (6 hospitals, Effective Bed Capacity)
   - Row 35–40: Net patient flow percentages
   - Rows 42–47: Min, Max, Median, Mean, 25th percentile, 75th percentile
   - Row 50: Weighted mean for MHN

3. **Read Data sheet**: The source data is in rows 21–38 of sheet `Data`. Build a lookup dictionary mapping `(series_code, year) -> value`. Identify which column holds the series codes and which columns hold the year values. Inspect the Data sheet structure first — print out a few rows to understand the layout before coding the lookup.

4. **Step 1 — Populate H12:L17, H19:L24, H26:L31**: For each cell in these ranges, get the series code from column D of that row on the Task sheet, get the year from row 10 of that column, look up the value from the Data sheet dictionary, and write the numeric value.

5. **Step 2 — Net patient flow (H35:L40)**: For each hospital (rows 35–40) and each year column (H–L), compute:
   `(Patient Admissions - Patient Discharges) / Effective Bed Capacity * 100`
   where Patient Admissions comes from the first block (rows 12–17), Patient Discharges from the second block (rows 19–24), and Effective Bed Capacity from the third block (rows 26–31). Match hospitals by their relative position (row offset 0–5 within each block).

6. **Step 2 — Summary statistics (H42:L47)**: For each year column, compute column-wise statistics over the 6 net-patient-flow values (rows 35–40):
   - Row 42: Minimum
   - Row 43: Maximum
   - Row 44: Median
   - Row 45: Simple mean (arithmetic average)
   - Row 46: 25th percentile
   - Row 47: 75th percentile
   Use `numpy.percentile` with default linear interpolation for percentiles.

7. **Step 3 — Weighted mean (H50:L50)**: For each year column, compute the weighted mean of the net patient flow values (H35:L40) using the Effective Bed Capacity values (H26:L31) as weights: `sum(flow_i * capacity_i) / sum(capacity_i)`.

8. **Verify row labels**: Before writing, print the labels in column B or C for the relevant rows to confirm the mapping (which rows are admissions, discharges, capacity, and which summary stat rows are min/max/median/mean/p25/p75). Adjust if the actual labels differ from the assumed order.

9. **Save**: Save the workbook to `/root/output/result.xlsx`. Do NOT add sheets, macros, or change formatting.

10. **Validate**: Reopen the saved file, read back a sample of the written cells, and print them to confirm they contain numeric values and are non-zero/non-None.

IMPORTANT: Inspect the actual sheet contents before writing. Print the series codes, year headers, and Data sheet structure to confirm assumptions. If anything differs from expectations, adapt accordingly.

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