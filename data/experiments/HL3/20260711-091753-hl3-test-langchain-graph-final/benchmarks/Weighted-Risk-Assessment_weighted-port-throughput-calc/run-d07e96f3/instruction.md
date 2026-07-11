# Task Instruction

You must update the Excel workbook `/root/data/workbook.xlsx` by inserting spreadsheet formulas (no VBA, no macros, no new sheets) and save the result to `/root/output/result.xlsx`.

## Preliminary Inspection

1. `mkdir -p /root/output`
2. Open and inspect the workbook with openpyxl. Print:
   - Sheet names.
   - On sheet `Task`: the contents of rows 10-50 for columns D through L (focus on column D for series codes, row 10 for years, and any existing content in the target ranges). Also print rows 4-8 to understand any headers/labels.
   - On sheet `Data`: rows 21-38, all columns, to understand the data layout (which column has the series code, which row/column has years, and where the values are).
   - Print the exact series codes in column D for rows 12-17, 19-24, 26-31, and 35-40 on sheet `Task`.
   - Print the exact years in cells H10:L10 on sheet `Task`.
   - Print the data block on sheet `Data` rows 21-38 to understand the lookup table structure (identify which column holds the series code, which row holds years, and where numeric data lives).

## Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas

Based on the inspection, write `INDEX/MATCH` formulas into each yellow cell. The pattern for each cell should be:

```
=INDEX(Data!<data_range>, MATCH($D<row>, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```

Adjust the exact ranges based on what you find in the Data sheet. The series code comes from column D of the current row on `Task`, and the year comes from row 10 of the current column on `Task`. The lookup source is `Data` rows 21-38.

IMPORTANT: Use absolute row references for row 10 (`H$10`, `I$10`, etc.) and absolute column references for column D (`$D12`, `$D13`, etc.) so the formula anchors correctly, but adapt the exact references to what the data layout requires.

Fill all 18 cells in each of the three blocks (6 rows × 5 columns each = 90 cells total).

## Step 2: Net container flow in H35:L40 and statistics in H42:L47

For each cell in H35:L40, the formula is:
```
=(H12 - H19) / H26 * 100
```
where row 12 corresponds to Loaded Containers Inbound, row 19 to Loaded Containers Outbound, and row 26 to Terminal Throughput Capacity. Adjust the row offsets so that the first port in row 35 uses rows 12, 19, 26; the second port in row 36 uses rows 13, 20, 27; etc.

Verify by checking that the series codes in rows 35-40 correspond to the same ports as rows 12-17 (and 19-24, 26-31) in order.

For H42:L47, compute column-wise statistics. Use these functions (use legacy names to avoid #NAME? errors in the openpyxl/calc engine):
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile — use PERCENTILE not PERCENTILE.INC)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile — use PERCENTILE not PERCENTILE.EXC)

Check the labels in column D (or nearby) for rows 42-47 to confirm the correct order of min/max/median/mean/25th/75th. Adjust row assignments if the labels indicate a different order.

## Step 3: Weighted mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the net container flow percentages using Terminal Throughput Capacity as weights.

## Final Steps

1. After inserting all formulas, save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and verify:
   - The formulas exist in the expected cells (spot-check a few from each block).
   - No cells are empty where formulas should be.
   - No new sheets were added.
   - The file loads without errors.
3. Print a summary of what was done.

## Critical Notes
- Do NOT use `.EXC` or `.INC` suffixed function names (e.g., use `PERCENTILE` not `PERCENTILE.INC`).
- Do NOT add any sheets, macros, VBA, or external links.
- Do NOT alter existing formatting.
- Use openpyxl for all operations.
- When writing formulas with openpyxl, assign them as strings starting with `=` to cell `.value`.
- Carefully inspect the Data sheet layout before writing any formulas — the exact column/row arrangement determines the INDEX/MATCH ranges.

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