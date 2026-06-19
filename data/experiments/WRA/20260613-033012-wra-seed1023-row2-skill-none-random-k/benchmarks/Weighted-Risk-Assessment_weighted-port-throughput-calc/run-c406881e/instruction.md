# Task Instruction

## Task: Update /root/data/workbook.xlsx with formulas and save to /root/output/result.xlsx

### Phase 0: Inspect the workbook
1. `mkdir -p /root/output`
2. Use `openpyxl` to open `/root/data/workbook.xlsx` (with `data_only=False` so you see formulas).
3. Read and print:
   - Sheet names.
   - Sheet `Task`: all content in rows 1–55, columns A–M. Pay special attention to:
     - Column D rows 12–17, 19–24, 26–31 (series codes for each block).
     - Row 10 columns H–L (years).
     - Row 35–40 column D (port names / series codes for Net container flow).
     - Rows 42–47 column G or nearby (labels: min, max, median, mean, 25th, 75th percentile).
     - Row 50 (CPA weighted mean).
   - Sheet `Data`: rows 21–38, all populated columns. Identify the layout: which row has headers, which column has series codes, how years are arranged (row-wise or column-wise).
4. Print the exact cell values so you understand the data layout before writing any formulas.

### Phase 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas
For each cell in these three 6×5 blocks:
- The lookup key is the series code in column D of the same row.
- The second input is the year in row 10 of the same column (H10, I10, ... L10).
- The source data is on sheet `Data` in rows 21:38.
- Use INDEX/MATCH (preferred) or VLOOKUP with MATCH. The exact formula depends on the Data sheet layout you discovered in Phase 0.
- Example pattern (adjust based on actual layout): `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))` — but you MUST adjust column/row references to match the actual Data sheet structure.
- Use absolute references for the data range and series-code column; use mixed references ($D12 for row-relative series code, H$10 for column-relative year).

### Phase 2: Net container flow in H35:L40
Based on the three blocks:
- Block 1 (H12:L17) = one metric (e.g., Loaded Containers Inbound)
- Block 2 (H19:L24) = another metric (e.g., Loaded Containers Outbound)
- Block 3 (H26:L31) = Terminal Throughput Capacity

Verify which block is which by checking the labels on the Task sheet (likely in column A or nearby for rows 11, 18, 25).

For each cell in H35:L40, the formula is:
`=(InboundCell - OutboundCell) / CapacityCell * 100`

where InboundCell, OutboundCell, and CapacityCell are the corresponding cells from the three blocks above (same column, corresponding row within each block — row offset 0 maps port 1, offset 1 maps port 2, etc.).

Verify that the ports in rows 35–40 match the order in rows 12–17 / 19–24 / 26–31. If they differ, match by port name or series code, not by position.

### Phase 3: Summary statistics in H42:L47
For each column H through L:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile) — or `PERCENTILE.INC`
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile) — or `PERCENTILE.INC`

Check the labels in column G (or wherever) for rows 42–47 to confirm the correct order of statistics. Adjust row assignments if the labels differ from the order above.

### Phase 4: Weighted mean in H50:L50
For each column (e.g., column H):
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the Net container flow percentages as values and Terminal Throughput Capacity as weights.

### Phase 5: Save and validate
1. Save the workbook to `/root/output/result.xlsx` preserving all existing formatting (do NOT change fonts, fills, borders, number formats, sheet structure).
2. Re-open `/root/output/result.xlsx` with `data_only=False` and print all formula cells you wrote to confirm they are present and correctly structured.
3. Also open with `data_only=True` (if supported / after a calc) to spot-check that formulas don't produce obvious errors.

### Critical constraints
- Do NOT add or remove any sheets.
- Do NOT add macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting (cell fills, fonts, borders, number formats on pre-existing cells).
- Only write into the specified cell ranges.
- Use `openpyxl` for all operations. When writing formulas, assign them as strings starting with `=`.
- If the Data sheet layout doesn't match assumptions, adapt the formulas accordingly — the key requirement is INDEX/MATCH or VLOOKUP/MATCH or HLOOKUP/MATCH or XLOOKUP/MATCH pattern using series code + year.

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