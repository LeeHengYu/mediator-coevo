# Task Instruction

## Task: Update /root/data/workbook.xlsx with formulas and save to /root/output/result.xlsx

### Phase 0: Inspect the workbook
1. `mkdir -p /root/output`
2. Use `openpyxl` to open `/root/data/workbook.xlsx` (with `data_only=False` so you see formulas).
3. Read and print the full contents of the `Task` sheet — every cell from row 1 through at least row 55, columns A–L. Pay special attention to:
   - Column D rows 12–17, 19–24, 26–31 (series codes for each block)
   - Row 10 columns H–L (the year headers)
   - Row 35–40 column D (port names or series codes for the Net container flow block)
   - Rows 42–47 column G or nearby (labels: min, max, median, mean, 25th, 75th percentile)
   - Row 50 (CPA weighted mean row)
   - Any existing formulas or values already present
4. Read and print the `Data` sheet rows 1–40, focusing on:
   - Row 21–38: the source data layout
   - The header row for the Data block (which row contains series codes? which row/column contains years?)
   - Determine the exact orientation: are series codes in a column and years across a row, or vice versa?

Print all of this before making any edits. You need to understand the exact layout.

### Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write a spreadsheet formula (not a Python-computed value) using one of the allowed lookup patterns. The formula must reference:
- The series code in column D of the **same row** on the `Task` sheet
- The year in row 10 of the **same column** on the `Task` sheet
- The source data on sheet `Data` rows 21:38

Choose the lookup pattern based on the Data sheet orientation:
- If Data has series codes in a column and years in a row header → `INDEX(MATCH, MATCH)` is cleanest
- Use absolute references (`$`) appropriately so the formula anchors the lookup range but allows the row's series code and column's year to vary.

Example pattern (adapt to actual layout): `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`

IMPORTANT: Use the actual column/row references you discovered in Phase 0. Do NOT guess.

### Phase 2: Net container flow in H35:L40

For each cell in H35:L40, write a formula that computes:
`(Loaded Containers Inbound - Loaded Containers Outbound) / Terminal Throughput Capacity * 100`

The three input blocks are:
- H12:L17 = one metric (check which one from the series codes — likely one of the three blocks corresponds to Inbound, Outbound, or Capacity)
- H19:L24 = another metric
- H26:L31 = another metric

Map them correctly based on the series codes you read in Phase 0. The order of ports in rows 35–40 should match the order in the source blocks (rows 12–17, 19–24, 26–31). Verify the port ordering matches.

Formula pattern: `=(H12 - H19) / H26 * 100` (adjust cell references based on which block is Inbound, Outbound, Capacity, and ensure row alignment with the ports in rows 35–40).

### Phase 3: Summary statistics in H42:L47

For each column H through L, write column-wise formulas:
- Row 42: `=MIN(H35:H40)` (or whichever label is in that row — check Phase 0)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

IMPORTANT: Match each formula to the actual label in column G (or wherever labels are). Read the labels first. The order might differ from what I listed above.

### Phase 4: Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the Net container flow percentages (H35:H40) as values and Terminal Throughput Capacity (H26:L31) as weights. Verify H26:L31 is indeed the Capacity block.

### Phase 5: Save and validate
1. Save the workbook to `/root/output/result.xlsx` preserving all existing formatting. Do NOT modify any cells outside the specified ranges.
2. Re-open the saved file and print all formula cells you wrote to confirm they were saved correctly.
3. Also open with `data_only=True` to check if openpyxl can at least parse without errors (values will show as None since no calc engine ran, but structure should be intact).

### Critical constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify existing formatting (fonts, colors, borders, etc.).
- Only write formulas into the specified yellow cells.
- All formulas must be Excel-compatible spreadsheet formulas, not Python-computed values.
- Use `openpyxl` for all operations. Load with `data_only=False` to preserve existing formulas elsewhere.

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