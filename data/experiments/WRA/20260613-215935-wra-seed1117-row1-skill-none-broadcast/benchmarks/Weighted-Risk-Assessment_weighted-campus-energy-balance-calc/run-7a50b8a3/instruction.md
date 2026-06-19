# Task Instruction

## Task: Weighted Campus Energy Balance Calculation

You must update `/root/data/workbook.xlsx` by populating specific cells with spreadsheet formulas, then save the result to `/root/output/result.xlsx`.

### Phase 0: Inspect the workbook
1. `mkdir -p /root/output`
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: read row 10 (the year headers in columns H–L), column D rows 12–31 (the series codes), rows 35–50 structure, and any existing content/formatting in the yellow target cells. Note the exact text in D12:D17, D19:D24, D26:D31 (series codes) and the campus names in the relevant rows.
   - Sheet `Data`: read rows 21–38 to understand the data layout. Determine: what is in row 20 (likely headers)? What column holds the series codes? What row holds the years? This is critical for choosing the right lookup formula structure.
   - Print all of this information so you understand the exact structure before writing any formulas.

### Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in the three blocks (rows 12–17, 19–24, 26–31, columns H–L), write a spreadsheet formula that:
- Uses the series code from column D of that row and the year from row 10 of that column
- Looks up the value from sheet `Data` rows 21:38
- Uses one of the approved patterns: `INDEX/MATCH`, `VLOOKUP/MATCH`, `HLOOKUP/MATCH`, or `XLOOKUP/MATCH`

IMPORTANT: You must write these as Excel *formulas* (strings starting with `=`) into the cells, NOT computed Python values. Use `openpyxl` to set the cell `.value` to a formula string.

When constructing the formula, pay careful attention to:
- Whether the Data sheet has series codes in a column (for VLOOKUP) or a row (for HLOOKUP)
- The exact range references needed
- Use absolute references where appropriate (e.g., `$D12` for the series code column, `H$10` for the year row)
- The sheet reference syntax: `Data!` prefix for ranges on the Data sheet

A good pattern if series codes are in column A of Data and years are in row 20 of Data:
```
=INDEX(Data!$A$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$A$20:$XX$20, 0))
```
Adjust the exact ranges based on what you discover in Phase 0. Make sure the ranges are correct.

### Phase 2: Net Renewable Balance in H35:L40

For each cell in H35:L40, write a formula computing:
```
(Renewable Generation - Grid Consumption) / Baseline Energy Demand * 100
```
where:
- Renewable Generation values are in H12:L17 (rows 12–17)
- Grid Consumption values are in H19:L24 (rows 19–24)  
- Baseline Energy Demand values are in H26:L31 (rows 26–31)

So for cell H35: `=(H12-H19)/H26*100`
For cell H36: `=(H13-H20)/H27*100`
etc. — match the row offsets so each campus row aligns correctly.

Verify the row mapping by checking that the campus names in rows 35–40 match those in rows 12–17, 19–24, 26–31. If the order differs, adjust accordingly.

### Phase 3: Summary statistics in H42:L47

For each column H–L, in rows 42–47, write formulas for:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

Check the labels in column D or nearby columns for rows 42–47 to confirm which row is which statistic. Adjust the row assignments to match the actual labels.

### Phase 4: Weighted mean in H50:L50

For each column (H–L), write a SUMPRODUCT formula:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net Renewable Balance percentages weighted by Baseline Energy Demand.

### Phase 5: Save and validate
1. Save the workbook to `/root/output/result.xlsx` preserving all existing formatting. Do NOT use `data_only=True` when loading. When saving, do not change any sheet properties or add sheets.
2. Re-open the saved file and verify:
   - Cells H12, H19, H26 contain formula strings (start with `=`)
   - Cells H35, H42, H50 contain formula strings
   - No new sheets were added
   - The formulas reference the correct ranges
3. Print a summary of sample formulas from each block for verification.

### Critical constraints
- Do NOT compute values in Python and write numbers. Write EXCEL FORMULAS.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify existing formatting (fonts, colors, borders, etc.).
- Do NOT use `data_only=True` when opening the workbook.
- Use `openpyxl` for all workbook operations.

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