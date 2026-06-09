# Task Instruction

Execute the following steps precisely to complete the hospital capacity workbook task.

## Setup
1. `cp /root/data/workbook.xlsx /root/output/result.xlsx`
2. Install openpyxl if needed: `pip install openpyxl`
3. Open `/root/output/result.xlsx` with openpyxl, preserving styles and formatting.

## Inspection
4. Read the `Task` sheet carefully:
   - Print the contents of rows 1–55, columns A–L, to understand the layout.
   - Identify what is in column D for rows 12–17, 19–24, 26–31 (series codes).
   - Identify what is in row 10 for columns H–L (years).
   - Identify what is in rows 35–40 (cluster names or references for Net capacity headroom).
   - Identify what is in rows 42–47 (labels: min, max, median, mean, 25th pctl, 75th pctl).
   - Identify what is in row 50 (Regional Care Grid weighted mean).
5. Read the `Data` sheet:
   - Print rows 1–40, focusing on rows 21–38, to understand the data layout.
   - Determine the structure: which row/column has series codes, which has years, which has values.
   - Note the exact row numbers and column layout so formulas reference correctly.

## Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas
6. For each cell in the three blocks (H12:L17, H19:L24, H26:L31):
   - The formula must use TWO inputs: (a) the series code from column D of that row, and (b) the year from row 10 of that column.
   - The lookup source is `Data!` rows 21:38.
   - Use INDEX/MATCH (most reliable in openpyxl). The pattern should be something like:
     `=INDEX(Data!<value_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))`
   - Adjust the exact ranges based on what you discover in the inspection step. The series codes column and year row on the Data sheet must be identified precisely.
   - Use absolute row references for the year row (e.g., `H$10`) and absolute column references for the series code column (e.g., `$D12`) so the formula copies correctly across the block.
   - Write the formulas using openpyxl by setting `cell.value = '=INDEX(...)'` as a string.

## Step 2: Net capacity headroom in H35:L40
7. For each cell in H35:L40 (6 rows × 5 columns):
   - The formula is: `(Available Care Slots - Occupied Care Slots) / Staffed Bed Capacity * 100`
   - From the inspection, determine which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to Available Care Slots, Occupied Care Slots, and Staffed Bed Capacity.
   - For example, if H12:L17 = Available Care Slots, H19:L24 = Occupied Care Slots, H26:L31 = Staffed Bed Capacity, then for cell H35: `=(H12-H19)/H26*100`
   - The row offset within each block should correspond (row 35 uses row 12, 19, 26; row 36 uses row 13, 20, 27; etc.).

8. For H42:L47, calculate column-wise statistics over H35:L40:
   - Identify which row label corresponds to which statistic (min, max, median, mean, 25th percentile, 75th percentile).
   - Use the exact labels from the sheet. Map them to Excel functions:
     - Minimum: `=MIN(H35:H40)`
     - Maximum: `=MAX(H35:H40)`
     - Median: `=MEDIAN(H35:H40)`
     - Simple mean: `=AVERAGE(H35:H40)`
     - 25th percentile: `=PERCENTILE(H35:H40,0.25)`
     - 75th percentile: `=PERCENTILE(H35:H40,0.75)`
   - Assign each formula to the correct row based on the label in column A/D/E of that row.

## Step 3: Weighted mean in H50:L50
9. For each cell in H50:L50:
   - `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
   - This computes the weighted mean of the Net capacity headroom percentages (H35:H40) weighted by Staffed Bed Capacity (H26:H31).
   - Adjust column references for each column (H, I, J, K, L).

## Save and Validate
10. Save the workbook (keep_vba=False is fine since no VBA). Close it.
11. Re-open the saved file and verify:
    - All formula cells in the target ranges contain formula strings (start with '=').
    - No cells are accidentally blank or contain plain values where formulas should be.
    - The sheet names are still `Task` and `Data` (no extra sheets).
    - Print a sample of formulas from each block to confirm correctness.
12. Do NOT add any new sheets, macros, VBA, external links, or helper tabs.

## Critical Notes
- Before writing ANY formula, complete the full inspection of both sheets. The exact row/column references in the Data sheet are essential.
- Use `data_only=False` when loading so existing formulas are preserved as formulas.
- Do NOT modify any cells outside the specified ranges.
- Do NOT change formatting, styles, or existing content.
- If the Data sheet has series codes in a column (e.g., column A or B) and years in a row (e.g., row 21 or row 20), adapt the INDEX/MATCH formula accordingly.
- Double-check that the PERCENTILE function arguments use 0.25 and 0.75 (not 25 and 75).

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