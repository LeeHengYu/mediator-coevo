# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Phase 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Use openpyxl to open `/root/data/workbook.xlsx` (data_only=False).
3. Print the sheet names to confirm `Task` and `Data` exist.
4. On sheet `Data`: print rows 19-40 (columns A-M) to understand the layout — specifically identify where series codes live (expected: column A) and where years live (expected: row 21, starting around column B or H). Print enough to see the exact column letters that hold year headers and the row numbers that hold data.
5. On sheet `Task`: print rows 9-52 (columns A-M). Identify:
   - Row 10: year headers in columns H-L.
   - Column D rows 12-17, 19-24, 26-31: series codes.
   - Rows 35-40: region names (and which metric blocks they correspond to).
   - Rows 42-47: labels (Min, Max, Median, Average/Mean, 25th percentile, 75th percentile) — read the exact text in column D/E/F/G.
   - Row 50: label for GCM weighted mean.
6. Print the yellow-highlighted cells' current content (H12:L17, H19:L24, H26:L31) to confirm they are empty or placeholder.

## Phase 1 – Lookup formulas (H12:L17, H19:L24, H26:L31)
For each cell in these three blocks, write an INDEX/MATCH formula:
```
=INDEX(Data!$B$21:$Z$38, MATCH($D{row}, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
Adjust the ranges based on what you discovered in Phase 0:
- The row-lookup array must be the column on `Data` that contains the series codes (likely column A, rows 21:38).
- The column-lookup array must be the row on `Data` that contains the year headers (likely row 20 or 21, determine from inspection).
- The data array must span from the first data column to the last data column, rows 21:38.
- Use absolute references for the data array and lookup arrays; use $D{row} (mixed: column absolute, row relative to current row) and {col}$10 (mixed: column relative, row absolute) so the formula copies correctly across the 5 columns and 6 rows of each block.

IMPORTANT: Use only standard Excel function names: INDEX, MATCH. Do NOT use XLOOKUP or any function that might cause #NAME? errors in older Excel or openpyxl evaluation.

## Phase 2 – Net reliability gap (H35:L40)
For each cell in H35:L40, write a formula:
```
=(H12 - H19) / H26 * 100
```
where the row offsets correspond to:
- H12:L17 = Successful API Requests block (rows 12-17)
- H19:L24 = Failed API Requests block (rows 19-24)
- H26:L31 = Compute Capacity block (rows 26-31)
So for row 35 col H: `=(H12-H19)/H26*100`, for row 36 col H: `=(H13-H20)/H27*100`, etc.
Verify the region ordering matches between all four blocks.

## Phase 3 – Summary statistics (H42:L47)
Based on the exact labels in column D/E/F/G for rows 42-47, assign formulas. Expected mapping (verify against actual labels):
- Minimum: `=MIN(H35:H40)` (or the equivalent column)
- Maximum: `=MAX(H35:H40)`
- Median: `=MEDIAN(H35:H40)`
- Mean/Average: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)`

CRITICAL: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid #NAME? errors. Similarly use `MEDIAN`, `MIN`, `MAX`, `AVERAGE` — all standard names.

For each column H through L, apply the same pattern referencing that column's H35:H40 (or I35:I40, etc.).

## Phase 4 – Weighted mean (H50:L50)
For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net reliability gap percentages using Compute Capacity as weights.

## Phase 5 – Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file with openpyxl (data_only=False).
3. For every cell in H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50, print the cell value and confirm it is a string starting with '=' (i.e., a formula).
4. Spot-check that no formula contains function names like XLOOKUP, PERCENTILE.INC, PERCENTILE.EXC, or any other potentially unsupported function.
5. Confirm no new sheets were added.
6. Print 'ALL DONE – formulas verified' when complete.

## Key Warnings
- Do NOT use `PERCENTILE.INC` or `PERCENTILE.EXC` — use `PERCENTILE` only.
- Do NOT use `XLOOKUP` — use `INDEX`/`MATCH` only.
- Do NOT modify formatting, add sheets, or add macros.
- Carefully inspect the Data sheet structure before writing any formulas — the exact row/column ranges are critical.
- If the year headers on Data are in a different row than expected, adjust all formulas accordingly.

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