# Task Instruction

Execute the following steps in order:

1. **Inspect the workbook layout.** Open `/root/data/workbook.xlsx` with openpyxl. On sheet `Task`, read:
   - The series codes in column D for rows 12–17, 19–24, 26–31 (three blocks of six regions).
   - The years in row 10 for columns H–L.
   - The labels/content of rows 35–50 to confirm the Net reliability gap block layout, the summary stats rows (42–47), and the weighted mean row (50).
   On sheet `Data`, read rows 21–38 to understand the data layout (which column holds the series code, which row holds the header with years, how the data is arranged).
   Print all of this so you have the exact cell references before writing any formulas.

2. **Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31.**
   For each cell in these three 6×5 blocks, write an `INDEX`/`MATCH` formula that:
   - Uses the series code from column D of the current row (e.g., `$D12` for row 12, with the `$` on the column so it stays fixed when copied across columns).
   - Uses the year from row 10 of the current column (e.g., `H$10` for column H, with the `$` on the row).
   - Looks up against the Data sheet rows 21:38. Identify the exact column that holds series codes and the exact row that holds years on the Data sheet from your inspection.
   - Pattern: `=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))`
   Adjust the ranges based on what you found in step 1. Make sure the data range, series code column, and year row are all correctly identified.

3. **Step 2 – Net reliability gap in H35:L40.**
   From your inspection, identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to:
   - Successful API Requests
   - Failed API Requests
   - Compute Capacity
   Then for each cell in H35:L40 (6 regions × 5 years), write:
   `=(H12-H19)/H26*100`  (adjust row references to match the correct blocks and current row offset)
   That is: `(Successful_cell - Failed_cell) / Capacity_cell * 100`

4. **Step 2 – Summary statistics in H42:L47.**
   For each column H through L:
   - Row 42 (Minimum): `=MIN(H35:H40)`
   - Row 43 (Maximum): `=MAX(H35:H40)`
   - Row 44 (Median): `=MEDIAN(H35:H40)`
   - Row 45 (Mean): `=AVERAGE(H35:H40)`
   - Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`  ← use legacy `PERCENTILE`, NOT `PERCENTILE.INC` or `PERCENTILE.EXC`
   - Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`  ← same, use legacy `PERCENTILE`

5. **Step 3 – Weighted mean in H50:L50.**
   For each column H through L:
   `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
   This uses the Net reliability gap values as the values and Compute Capacity as the weights.

6. **Save** the workbook to `/root/output/result.xlsx`. Create the `/root/output/` directory if it doesn't exist. Use `openpyxl` to save, preserving existing formatting.

7. **Validate.** Re-open `/root/output/result.xlsx` and spot-check:
   - That cells H12, L17, H19, L24, H26, L31 contain formulas (start with `=`).
   - That H35 contains a formula referencing the three blocks.
   - That H46 contains `=PERCENTILE(...)` (not PERCENTILE.INC).
   - That H50 contains a SUMPRODUCT formula.
   - Print a few formula strings to confirm correctness.

**Critical constraints:**
- Use `PERCENTILE` (legacy), never `PERCENTILE.INC` or `PERCENTILE.EXC`.
- Do not add sheets, macros, VBA, external links, or helper tabs.
- Do not alter existing formatting.
- All formulas must be spreadsheet formulas (strings starting with `=`), not hardcoded values.

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