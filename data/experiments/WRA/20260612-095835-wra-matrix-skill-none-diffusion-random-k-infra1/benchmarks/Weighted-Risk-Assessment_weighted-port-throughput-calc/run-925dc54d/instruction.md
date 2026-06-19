# Task Instruction

Execute the following steps to complete the task. Read each step carefully before acting.

## 0. Setup and Inspection

1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Open `/root/output/result.xlsx` with openpyxl (with `data_only=False` so you can read and write formulas).
3. Inspect the `Task` sheet thoroughly:
   - Print the contents of rows 10–50, columns A–L, to understand the layout.
   - Identify what is in column D for rows 12–17, 19–24, 26–31, 35–40 (these should be series codes).
   - Identify what is in row 10 for columns H–L (these should be years).
   - Print the contents of rows 42–47 column A–G to see the stat labels (min, max, median, mean, 25th, 75th percentile).
   - Print row 50 columns A–G to see the CPA label.
4. Inspect the `Data` sheet:
   - Print rows 21–38 fully (all columns with data) to understand the data layout: which row has which series, which column has which year, etc.
   - Determine the structure: Are series codes in a column? Are years in a row? Which column contains the series codes? Which row contains the years?
   - Note the exact column letters and row numbers.

## 1. Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas

For each cell in the ranges H12:L17, H19:L24, and H26:L31:
- The lookup key 1 is the series code in column D of the same row on sheet `Task`.
- The lookup key 2 is the year in row 10 of the same column on sheet `Task`.
- The data source is sheet `Data` rows 21:38.

Based on your inspection of the Data sheet, construct an appropriate formula using INDEX/MATCH (preferred) or another allowed pattern. The formula pattern should be something like:

`=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))`

Adjust the exact references based on what you found in the Data sheet:
- `<data_range>`: The rectangular range on Data sheet covering rows 21:38 and the data columns.
- `<series_code_column>`: The column on Data sheet that contains the series codes (likely column A or B), restricted to rows 21:38.
- `<year_row>`: The row on Data sheet that contains the year headers.

IMPORTANT: Use `$D12` (column locked) for the series code reference so it doesn't shift horizontally. Use `H$10` (row locked) for the year reference so it doesn't shift vertically. Adjust row references as you move through the ranges.

Write the formula into each cell. Do NOT overwrite any cells outside the specified yellow ranges.

## 2. Step 2a: Net container flow in H35:L40

The formula for each cell is:
`(Loaded Containers Inbound - Loaded Containers Outbound) / Terminal Throughput Capacity * 100`

Based on the Task sheet layout:
- Rows 12–17 correspond to one block (check what metric this is from labels in columns A–C).
- Rows 19–24 correspond to another block.
- Rows 26–31 correspond to another block.
- Rows 35–40 are for the six ports' Net container flow.

Identify which block is "Loaded Containers Inbound", which is "Loaded Containers Outbound", and which is "Terminal Throughput Capacity" by reading the labels. Then for each cell in H35:L40, write a formula like:
`=(H12-H19)/H26*100` (adjust row references to match the correct port in each block).

The ports in rows 35–40 should correspond to the same ports in the same order as rows 12–17, 19–24, and 26–31. Verify this by checking column D or the port names.

## 3. Step 2b: Summary statistics in H42:L47

For each column H through L, calculate column-wise statistics over the 6 values in rows 35–40:
- Row 42: `=MIN(H35:H40)` (or whichever row is minimum per the labels)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

IMPORTANT: Match the statistic to the label in each row. Read the labels in column A/B/C for rows 42–47 first, then assign formulas accordingly. The order above is just a guess.

## 4. Step 3: Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net container flow percentages (H35:H40) weighted by Terminal Throughput Capacity (H26:H31). Use SUMPRODUCT as required.

Lock row references appropriately if needed (e.g., `H$35:H$40` and `H$26:H$31`).

## 5. Save and Validate

1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the file and print the formula contents of all modified cells to verify:
   - H12:L17 contain INDEX/MATCH (or equivalent) formulas referencing Data sheet.
   - H19:L24 contain similar formulas.
   - H26:L31 contain similar formulas.
   - H35:L40 contain the net container flow calculation.
   - H42:L47 contain MIN, MAX, MEDIAN, AVERAGE, PERCENTILE formulas.
   - H50:L50 contain SUMPRODUCT formulas.
3. Verify no other cells were modified and no new sheets were added.
4. Open with data_only=True and check that the lookup formulas would resolve (they may show None in openpyxl but the formulas should be syntactically correct).

## Critical Notes
- Do NOT add any new sheets, macros, VBA, or external links.
- Do NOT change any existing formatting.
- Read the actual cell contents before and after every edit.
- If the Data sheet layout is different from expected, adapt the formulas accordingly based on what you actually see.
- Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) unless the labels specifically indicate otherwise. Actually, `PERCENTILE.INC` is the modern equivalent and is more commonly expected in .xlsx files — use whichever is standard. Check if the workbook already uses any percentile functions for guidance.
- Make sure the series code references in the lookup formulas use mixed references ($D12 pattern) so they work correctly across the H:L columns.

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