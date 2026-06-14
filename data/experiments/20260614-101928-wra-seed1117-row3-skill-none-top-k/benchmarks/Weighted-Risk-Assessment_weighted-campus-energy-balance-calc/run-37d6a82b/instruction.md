# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`:

1. **Inspect the workbook** – Open `/root/data/workbook.xlsx` with openpyxl and inspect:
   - Sheet `Task`: read row 10 (years in H10:L10), column D for series codes in rows 12-17, 19-24, 26-31. Note the exact text of each series code and each year header. Also read the labels in rows 35-40 (campus names), rows 42-47 (stat labels), and row 50 (MCEC label). Check what is in H12:L31 (should be empty/yellow), H35:L47, and H50:L50.
   - Sheet `Data`: read rows 21-38 to understand the data layout – identify which column holds the series code and which row/column holds each year's value. Print a few sample rows so you understand the orientation (is the lookup key in a column and years across columns, or vice-versa?).

2. **Populate H12:L17, H19:L24, H26:L31 with lookup formulas** – For each cell in these ranges, write a formula that combines INDEX and MATCH (or another approved pattern) to look up the value from `Data!$A$21:$Z$38` (adjust the exact range based on your inspection). The formula should:
   - Use the series code from column D of the same row on `Task` (e.g., `$D12`).
   - Use the year from row 10 of the same column on `Task` (e.g., `H$10`).
   - Example pattern: `=INDEX(Data!$B$21:$F$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$F$20, 0))` – but adjust column letters and row numbers to match what you actually find in the Data sheet. Lock references appropriately so the formula can be filled across the 5×6 blocks.
   - Write the formula string into each cell (do NOT compute the value in Python; write the Excel formula so it is a live formula in the workbook).

3. **Populate H35:L40 – Net renewable balance** – For each campus row i (rows 35-40 correspond to campuses in rows 12-17 / 19-24 / 26-31), write a formula:
   `=(H12 - H19) / H26 * 100`  (adjusted for the correct row offsets per campus and column). The three blocks are: Renewable Generation (rows 12-17), Grid Consumption (rows 19-24), Baseline Energy Demand (rows 26-31). So row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.

4. **Populate H42:L47 – Column-wise statistics** – For each column (H through L):
   - Row 42: `=MIN(H35:H40)`
   - Row 43: `=MAX(H35:H40)`
   - Row 44: `=MEDIAN(H35:H40)`
   - Row 45: `=AVERAGE(H35:H40)`
   - Row 46: `=PERCENTILE(H35:H40, 0.25)`
   - Row 47: `=PERCENTILE(H35:H40, 0.75)`
   Verify the stat labels in column D/E (or wherever they are) to confirm the order (min, max, median, mean, 25th, 75th). Adjust the row assignments if the labels differ.

5. **Populate H50:L50 – Weighted mean** – For each column:
   `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
   This uses the Net renewable balance percentages as values and Baseline Energy Demand as weights.

6. **Save** – Save the workbook to `/root/output/result.xlsx`. Create the `/root/output/` directory if it doesn't exist. Do NOT change formatting, do NOT add sheets.

7. **Validate** – Re-open the saved file with openpyxl (data_only=False) and confirm:
   - Cells H12, L17, H19, L24, H26, L31 contain formula strings (start with '=').
   - Cells H35, L40 contain formula strings.
   - Cells H42, L47 contain formula strings.
   - Cell H50 contains a formula string.
   - No new sheets were added.

**Critical notes from prior failures:**
- You MUST write Excel formula strings (e.g., `ws['H12'] = '=INDEX(...)'`), not computed Python values. Cells must contain live formulas.
- Inspect the Data sheet carefully before writing formulas – get the exact range boundaries right.
- Do not use `data_only=True` when writing; use the default mode.
- Preserve all existing formatting by not touching any cells outside the specified ranges.

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