# Task Instruction

Execute the following steps to produce /root/output/result.xlsx:

1. **Inspect the workbook** – Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Print:
   - Sheet names.
   - `Task` sheet: values in D12:D17, D19:D24, D26:D31 (series codes), and H10:L10 (year headers).
   - `Task` sheet: labels in G35:G40, G42:G47, G50.
   - `Data` sheet: row 21 headers (columns A–Z or until empty) and column A/B values for rows 21–38.
   - Note the exact column letters where years appear in `Data` row 21 and where series codes appear in `Data` column A or B.

2. **Populate H12:L17, H19:L24, H26:L31 with INDEX/MATCH formulas.**
   Each cell should use the pattern:
   ```
   =INDEX(Data!<data_range>, MATCH(<series_code_cell>, Data!<series_code_column>, 0), MATCH(<year_cell>, Data!<year_row>, 0))
   ```
   - `<data_range>`: the rectangular block on `Data` sheet rows 21–38 that contains both the row headers (series codes) and column headers (years). Determine exact references from inspection.
   - `<series_code_cell>`: absolute reference to column D of the current row on `Task` (e.g., `$D12`).
   - `<series_code_column>`: the column in `Data` that holds the series codes (likely column A or B, rows 21–38).
   - `<year_cell>`: absolute reference to the year in row 10 of `Task` (e.g., `H$10`).
   - `<year_row>`: the row on `Data` that holds the year headers (row 21), spanning the data columns.
   Use `0` for exact match in both MATCH calls. Lock references appropriately so the formula can be filled across columns and down rows.

3. **Net renewable balance (H35:L40).**
   For each campus (rows 35–40) and each year column (H–L):
   ```
   =(H12 - H19) / H26 * 100
   ```
   Adjust row references so row 35 uses rows 12, 19, 26; row 36 uses 13, 20, 27; etc.

4. **Summary statistics (H42:L47).**
   For each column (H–L):
   - Row 42 (Min):    `=MIN(H35:H40)`
   - Row 43 (Max):    `=MAX(H35:H40)`
   - Row 44 (Median): `=MEDIAN(H35:H40)`
   - Row 45 (Mean):   `=AVERAGE(H35:H40)`
   - Row 46 (25th percentile): `=_xlfn.PERCENTILE.INC(H35:H40,0.25)`
   - Row 47 (75th percentile): `=_xlfn.PERCENTILE.INC(H35:H40,0.75)`

   **Critical:** Use the prefix `_xlfn.` before `PERCENTILE.INC`. This is required by openpyxl so that Excel/evaluator recognises the function. Do NOT use bare `PERCENTILE.INC` or `PERCENTILE` without the prefix. Verify by reading back the cell values after writing.

5. **Weighted mean (H50:L50).**
   For each column (H–L):
   ```
   =SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
   ```
   This computes the weighted mean of the net renewable balance percentages using Baseline Energy Demand as weights.

6. **Save** the workbook to `/root/output/result.xlsx` (create `/root/output/` if needed). Do not add sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

7. **Validate** – Re-open the saved file with openpyxl (data_only=False). Print the formula strings in cells H12, L31, H35, H42, H46, H47, H50 to confirm they are correctly written and contain `_xlfn.PERCENTILE.INC` for rows 46–47.

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