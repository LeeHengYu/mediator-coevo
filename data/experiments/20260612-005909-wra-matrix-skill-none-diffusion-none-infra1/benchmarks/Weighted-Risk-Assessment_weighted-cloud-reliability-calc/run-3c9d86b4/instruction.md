# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 – Environment setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1 – Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Before writing any formulas, print:
- **Task sheet**: The contents of cells D12:D17, D19:D24, D26:D31 (series codes), row 10 columns H–L (year headers), cells H35:L40 labels, H42:H47 labels, row 50 label.
- **Data sheet**: Row 21 through 38 – print every row so you can see the exact layout (columns A through at least column AZ or wherever data ends). Identify: which column holds the series codes, which row holds the year headers, and the column/row ranges that contain the numeric data.

This inspection is **critical** – do NOT skip it. All subsequent formula construction depends on the exact layout.

## 2 – Write lookup formulas (Step 1)
Using the inspection results, write `INDEX(MATCH,MATCH)` formulas into the yellow cells:
- Blocks: H12:L17, H19:L24, H26:L31 (6 rows × 5 columns each = 90 formulas).
- Each formula pattern:
  ```
  =INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
  ```
  Replace `<data_range>`, `<series_code_column>`, and `<year_header_row>` with the exact ranges you discovered in the inspection. Use `$D12` (column-absolute) and `H$10` (row-absolute) so the formula copies correctly across the 5 columns and down the 6 rows of each block.
- Make sure the `<data_range>`, `<series_code_column>`, and `<year_header_row>` are consistent: the MATCH for the series code should return a row index into `<data_range>`, and the MATCH for the year should return a column index into `<data_range>`.

## 3 – Net reliability gap formulas (Step 2, rows 35–40)
For each of the 6 region rows (rows 35–40) and each year column (H–L):
```
= (H12 - H19) / H26 * 100
```
where H12 is from the first block (Successful API Requests), H19 from the second block (Failed API Requests), H26 from the third block (Compute Capacity). Adjust row references for each region row:
- Row 35 uses rows 12, 19, 26
- Row 36 uses rows 13, 20, 27
- Row 37 uses rows 14, 21, 28
- Row 38 uses rows 15, 22, 29
- Row 39 uses rows 16, 23, 30
- Row 40 uses rows 17, 24, 31

## 4 – Summary statistics (Step 2, rows 42–47)
For each year column (H–L), write these formulas referencing the Net reliability gap block H35:H40 (adjust column letter):
- Row 42 (MIN):    `=MIN(H35:H40)`
- Row 43 (MAX):    `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (AVERAGE/mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)`

**Important**: Check the labels in cells to the left of rows 42–47 during inspection to confirm the correct order (min, max, median, mean, 25th, 75th). Adjust the row assignments if the labels differ from the order above.

## 5 – Weighted mean (Step 3, row 50)
For each year column (H–L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net reliability gap percentages using Compute Capacity as weights.

## 6 – Save
Save the workbook to `/root/output/result.xlsx`. Do **not** add sheets, macros, VBA, external links, or helper tabs. Do not alter any existing formatting.

## 7 – Verification
After saving, reopen `/root/output/result.xlsx` with openpyxl and print the formulas (not values) in a sample of cells: H12, L17, H19, L24, H26, L31, H35, L40, H42, L47, H50, L50. Confirm they are non-empty formula strings (starting with '='). If any are None or empty, debug and fix before finishing.

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