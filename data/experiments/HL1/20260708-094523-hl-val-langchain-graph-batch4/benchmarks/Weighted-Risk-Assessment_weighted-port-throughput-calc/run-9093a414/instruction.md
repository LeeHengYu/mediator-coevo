# Task Instruction

You must update an Excel workbook at `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Follow these steps precisely:

## Step 0: Inspect the workbook
1. `mkdir -p /root/output`
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: print rows 1–55, focusing on columns A–L. Pay special attention to:
     - Row 10 (year headers in H10:L10)
     - Column D rows 12–17, 19–24, 26–31 (series codes)
     - The structure of H12:L17, H19:L24, H26:L31 (the yellow cells to fill)
     - Rows 35–40 (Net container flow rows — what ports, what structure)
     - Rows 42–47 (min, max, median, mean, 25th pctl, 75th pctl labels)
     - Row 50 (CPA weighted mean)
   - Sheet `Data`: print rows 21–38, all columns. Identify the layout — which row/column has series codes, which has years, etc.
3. Print cell values, formatting info (especially fill colors to confirm yellow cells), and any existing formulas.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write an Excel formula that:
- Takes the series code from column D of that row
- Takes the year from row 10 of that column (H10, I10, etc.)
- Looks up the value from sheet `Data` rows 21:38
- Uses one of the allowed patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH

Before writing formulas, carefully determine:
- The exact layout of the Data sheet (rows 21:38): Is the series code in column A? Are years in a header row? Which row is the header?
- Whether a horizontal or vertical lookup is more natural.
- Use absolute references for the Data range and MATCH ranges where appropriate, and relative/mixed references for the lookup values (series code, year) so formulas can be filled across the range.

A recommended approach (adjust based on actual Data layout):
- If Data has series codes in a column and years in a row header, use `INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))` — but adjust ranges to match actual data positions.

Write the formula into each cell. Do NOT set cell values to numbers; they must be formulas.

## Step 2: Net container flow and summary statistics

### H35:L40 — Net container flow
For each cell, write a formula:
`=(Loaded_Inbound - Loaded_Outbound) / Terminal_Throughput_Capacity * 100`

where:
- Loaded Containers Inbound values are in H12:L17 (the first block)
- Loaded Containers Outbound values are in H19:L24 (the second block)
- Terminal Throughput Capacity values are in H26:L31 (the third block)

So for H35: `=(H12-H19)/H26*100`, for H36: `=(H13-H20)/H27*100`, etc. Verify the row mapping by checking which port is in each row — the ports in rows 35–40 must match the ports in rows 12–17, 19–24, 26–31. If the port order differs, adjust the references accordingly.

### H42:L47 — Summary statistics (column-wise)
For each column (H through L):
- Row 42 (minimum): `=MIN(H35:H40)`
- Row 43 (maximum): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Check the labels in column A/B/C/D for rows 42–47 to confirm which statistic goes in which row. Match accordingly.

## Step 3: Weighted mean in H50:L50
For each column (H through L), write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of Net container flow percentages using Terminal Throughput Capacity as weights.

## Step 4: Save and Validate
1. Save to `/root/output/result.xlsx` using `openpyxl`, preserving all existing formatting. When opening the workbook, use `keep_vba=False` (default) and do NOT use `data_only=True` (you need to preserve formulas).
2. Re-open the saved file and verify:
   - Cells in H12:L17, H19:L24, H26:L31 contain formulas (not plain values)
   - Cells in H35:L40 contain formulas
   - Cells in H42:L47 contain formulas
   - Cells in H50:L50 contain formulas
   - No new sheets were added
   - Print a sample of formulas to confirm correctness

## Critical constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting (fonts, colors, borders, column widths, etc.).
- All yellow cells must contain Excel formulas, not hardcoded values.
- The lookup formulas MUST use one of: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.
- Use `openpyxl` for all Excel operations. If you need to write formulas, assign them as strings starting with `=` to the cell's `.value` property.
- Inspect the actual Data sheet layout carefully before writing any formulas. The exact column/row references depend on where series codes and year headers actually are.

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