# Task Instruction

## Task: Update /root/data/workbook.xlsx with formulas and save to /root/output/result.xlsx

### Phase 0: Inspect the workbook
1. `mkdir -p /root/output`
2. Use `openpyxl` to open `/root/data/workbook.xlsx` (with `data_only=False` so you see formulas).
3. Read sheet `Task`:
   - Print rows 10-50, columns A-L (values and any existing formulas), paying special attention to:
     - Row 10 (years in H10:L10)
     - Column D rows 12-17, 19-24, 26-31 (series codes)
     - Column A or B rows 12-17, 19-24, 26-31 (labels/port names)
     - Rows 35-40 (port names/labels for Net container flow)
     - Row 50 (CPA weighted mean row)
     - Rows 42-47 (min, max, median, mean, 25th, 75th percentile labels)
   - Print any existing content in the yellow cell ranges to understand what's pre-filled vs empty.
4. Read sheet `Data`:
   - Print rows 21-38 fully (all columns) to understand the data layout: where series codes are, where years appear, and how the data is structured (row-oriented vs column-oriented).
   - Identify which column contains the series codes and which row/column contains the year headers.

### Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on your inspection, construct formulas using `INDEX/MATCH` (preferred) or another allowed pattern. Each formula must:
- Reference the series code from column D of the current row (e.g., `$D12`)
- Reference the year from row 10 of the current column (e.g., `H$10`)
- Look up from sheet `Data` rows 21:38

The exact formula structure depends on the Data sheet layout:
- If Data has series codes in a column and years in a header row, use something like: `=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))`
- Adjust ranges based on actual inspection. Make sure the ranges are correct and cover rows 21-38 of the Data sheet.
- Use absolute row/column references (`$`) appropriately so formulas can be placed across the 5 columns (H-L) and multiple rows correctly.

Write these formulas into cells H12:L17, H19:L24, and H26:L31 on the `Task` sheet.

### Phase 2: Net container flow in H35:L40 and statistics in H42:L47

For H35:L40, determine which rows correspond to:
- Loaded Containers Inbound (should be in H12:L17 block)
- Loaded Containers Outbound (should be in H19:L24 block)  
- Terminal Throughput Capacity (should be in H26:L31 block)

The formula for each cell in H35:L40 is:
`=(H12-H19)/H26*100` (adjusting row references for each port, matching the same port across the three blocks)

Verify that the port ordering in rows 35-40 matches rows 12-17, 19-24, and 26-31. If the ordering differs, adjust references accordingly.

For H42:L47 (column-wise statistics over H35:L40):
- H42: `=MIN(H35:H40)` (minimum)
- H43: `=MAX(H35:H40)` (maximum)
- H44: `=MEDIAN(H35:H40)` (median)
- H45: `=AVERAGE(H35:H40)` (simple mean)
- H46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- H47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

Check the labels in column A/B/C for rows 42-47 to confirm the correct order of statistics. Adjust row assignments if the labels differ from the order above.

### Phase 3: Weighted mean in H50:L50

For each column (H through L):
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of Net container flow percentages using Terminal Throughput Capacity as weights.

### Phase 4: Save and validate
1. Save the workbook to `/root/output/result.xlsx` preserving all existing formatting. Use `openpyxl` and make sure NOT to use `data_only=True` when loading (so formulas are preserved). Do not modify any styles, formatting, or sheet structure.
2. Re-open the saved file and verify:
   - Cells H12:L17, H19:L24, H26:L31 contain formula strings (not None/empty)
   - Cells H35:L40 contain formula strings
   - Cells H42:L47 contain formula strings
   - Cells H50:L50 contain formula strings
   - Print all formula cells to confirm correctness
   - Confirm no extra sheets were added
   - Confirm sheets `Task` and `Data` still exist

### Important constraints
- Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting (fonts, colors, borders, etc.).
- Do NOT use `data_only=True` when loading the workbook for editing.
- All formulas must be Excel-compatible spreadsheet formulas (strings starting with `=`).
- The SUMPRODUCT formula in H50:L50 must explicitly use the SUMPRODUCT function as required by the task.
- Double-check every cell reference in formulas against the actual workbook layout before writing.

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