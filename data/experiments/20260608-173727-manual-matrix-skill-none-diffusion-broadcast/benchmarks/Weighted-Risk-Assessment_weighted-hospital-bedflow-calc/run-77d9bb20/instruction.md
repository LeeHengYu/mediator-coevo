# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
- Open `/root/data/workbook.xlsx` with openpyxl (keep formulas via `data_only=False`).
- Print the sheet names to confirm `Task` and `Data` exist.
- Print the contents of `Task` sheet rows 10-50, columns D through L, so you can see:
  - Row 10: the years in H10:L10
  - Column D rows 12-17, 19-24, 26-31: the series codes
  - Row 35-40: which hospitals correspond to Net patient flow
  - Rows 42-47: labels for min, max, median, mean, 25th, 75th percentile
  - Row 50: Metro Hospital Network weighted mean
- Print `Data` sheet rows 21-38 (all columns) to understand the data layout: which row holds which series code, which columns hold which years, and how the data is structured (row-oriented vs column-oriented).
- Also print `Data` sheet row 1 (or header rows) and column A for rows 1-40 to understand the full structure.

## 2. Determine lookup orientation
Based on the inspection:
- Identify whether series codes are in a column (for VLOOKUP/INDEX-MATCH) or a row (for HLOOKUP) in Data rows 21:38.
- Identify where years appear in the Data sheet (likely in a header row or column).
- Choose the appropriate lookup pattern. A robust universal approach is `INDEX(Data!$A$21:$Z$38, MATCH(D12, <series_code_range>, 0), MATCH(H$10, <year_range>, 0))` — adjust ranges based on actual layout.

## 3. Write formulas for Step 1 (H12:L17, H19:L24, H26:L31)
Using openpyxl, write formula strings into each cell. For each cell at row `r`, column `c` (H=8, I=9, J=10, K=11, L=12):
- The series code reference is `$D{r}` (column D of the current row).
- The year reference is `{col_letter}$10` (row 10 of the current column).
- Use INDEX-MATCH pattern like: `=INDEX(Data!$B$21:$Z$38, MATCH($D{r}, Data!$A$21:$A$38, 0), MATCH({col}$10, Data!$B$20:$Z$20, 0))`
- IMPORTANT: Adjust the exact ranges based on what you observed in step 1. The series codes might be in column A or B of Data; the years might be in row 20 or another header row. Get this right by inspecting the actual data.
- Make sure to use mixed references correctly: lock the series code column with `$D` and lock the year row with `$10`.

Write these formulas for all three blocks (rows 12-17, 19-24, 26-31), columns H through L.

## 4. Write formulas for Step 2 — Net patient flow (H35:L40)
For each hospital row in 35-40:
- Identify which rows contain Patient Admissions, Patient Discharges, and Effective Bed Capacity for that hospital. Based on the layout:
  - Rows 12-17 likely correspond to one metric (e.g., Patient Admissions)
  - Rows 19-24 likely correspond to another metric (e.g., Patient Discharges)
  - Rows 26-31 likely correspond to Effective Bed Capacity
- Verify by checking the labels in the Task sheet (column B or C).
- The formula for cell H35 should be something like: `=(H12-H19)/H26*100` — mapping the first hospital across the three blocks. Adjust row offsets for each hospital (rows 35→12/19/26, 36→13/20/27, etc.).

## 5. Write formulas for Step 2 — Summary statistics (H42:L47)
For each column (H through L), in the six rows 42-47, write:
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- MEAN: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- 75th percentile: `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`
- IMPORTANT: Check the labels in column B/C/D for rows 42-47 to determine which row gets which statistic. Map them correctly.

## 6. Write formulas for Step 3 — Weighted mean (H50:L50)
For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
The instruction says to use SUMPRODUCT with Step 2 percentages as values and Effective Bed Capacity as weights. This is the standard weighted mean formula.

## 7. Save and validate
- Save the workbook to `/root/output/result.xlsx`.
- Reopen the saved file and print the formula cells to confirm formulas were written (not just values).
- Verify no new sheets were added.
- Verify formatting was not changed (openpyxl preserves formatting when you only write to cells without touching styles).

## Critical Notes
- Do NOT use `data_only=True` when loading — you need to preserve existing formulas and write new ones.
- Do NOT add any new sheets, delete sheets, or modify cells outside the specified ranges.
- When writing formulas, always start the string with `=`.
- After inspecting the Data sheet layout, adapt all range references accordingly. Do not assume ranges — verify them from the actual file content.
- If the Data sheet has years in a row and series codes in a column, use INDEX(MATCH, MATCH). If transposed, adjust.
- Print intermediate results to confirm correctness at each step.

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