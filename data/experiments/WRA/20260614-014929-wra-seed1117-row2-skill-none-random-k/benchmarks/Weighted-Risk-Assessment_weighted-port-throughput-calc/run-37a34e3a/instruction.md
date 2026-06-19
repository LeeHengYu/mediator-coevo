# Task Instruction

## Task: Update /root/data/workbook.xlsx with formulas and save to /root/output/result.xlsx

### Phase 0: Inspect the workbook
1. `mkdir -p /root/output`
2. Use `openpyxl` to open `/root/data/workbook.xlsx` (with `data_only=False` to preserve formulas).
3. Print the sheet names. Confirm `Task` and `Data` exist.
4. Print the contents of sheet `Task` for rows 1–55, columns A–M, so you can see:
   - Column D (series codes) for rows 12–17, 19–24, 26–31, 35–40
   - Row 10 (years in columns H–L)
   - Row 42–47 labels (min, max, median, mean, 25th, 75th percentile)
   - Row 50 label
   - Any existing content/formulas in the yellow cells
5. Print sheet `Data` rows 19–40, all populated columns, to understand the data layout (column headers, row structure, series codes, where numeric data lives).
6. Carefully note:
   - The exact series codes in column D of Task sheet for each block
   - The exact structure of Data sheet rows 21:38 — which row has headers, which column has series codes, which columns/rows have year data
   - Whether Data is arranged with years in columns (suitable for VLOOKUP) or years in rows (suitable for HLOOKUP)

### Phase 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas

For each cell in these ranges, write a spreadsheet formula (as a string starting with `=`) that looks up the value from sheet `Data` rows 21:38.

**Choose the lookup pattern based on Data sheet layout:**
- If Data has series codes in a column and years across columns: use `INDEX(MATCH, MATCH)` or `VLOOKUP` with `MATCH` for the column.
- If Data has years in rows: use `HLOOKUP` with `MATCH` or `INDEX(MATCH, MATCH)`.

**Formula construction rules:**
- Each formula must use TWO inputs: (1) the series code from column D of the SAME row on Task sheet (e.g., `$D12`), and (2) the year from row 10 of the SAME column on Task sheet (e.g., `H$10`).
- Use absolute references appropriately: lock the column for D references (`$D12`) and lock the row for year references (`H$10`). Lock the data range fully (`$...$`).
- The lookup range must reference `Data!` rows 21:38.
- Use one of these patterns: `VLOOKUP`+`MATCH`, `HLOOKUP`+`MATCH`, `XLOOKUP`+`MATCH`, or `INDEX`+`MATCH`.
- Make sure the formula is correct for the actual Data sheet layout you observed.

Write the formula for cell H12 first, verify it looks correct by inspecting the Data sheet structure, then apply the analogous formula to all cells in H12:L17, H19:L24, H26:L31 (adjusting only row references naturally through the pattern of `$D{row}` and `{col}$10`).

### Phase 2: Net container flow in H35:L40

For each of the 6 ports (rows 35–40) and 5 years (columns H–L):
- Identify which rows in the Task sheet contain "Loaded Containers Inbound" (rows 12–17), "Loaded Containers Outbound" (rows 19–24), and "Terminal Throughput Capacity" (rows 26–31).
- Confirm the port ordering is the same across all three blocks and the H35:L40 block. If port in row 35 corresponds to row 12, 19, and 26, etc.
- Formula: `=(H12 - H19) / H26 * 100` (adjusted for each row/column pair). Use relative references so they naturally adjust.

### Phase 3: Summary statistics in H42:L47

For each column H–L, in rows 42–47, write formulas for the column-wise statistics over H35:L40 (the 6 Net container flow values):
- Row 42 (minimum): `=MIN(H35:H40)`
- Row 43 (maximum): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

**IMPORTANT:** Check the actual labels in column D/E for rows 42–47 to determine which row gets which statistic. Match the label to the formula. Do NOT assume the order above — read the labels first.

### Phase 4: Weighted mean in H50:L50

For each column H–L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the Net container flow percentages as values and Terminal Throughput Capacity as weights.

### Phase 5: Save and Validate
1. Save the workbook to `/root/output/result.xlsx` using openpyxl. Do NOT use `data_only=True` when loading — preserve all formulas.
2. Re-open `/root/output/result.xlsx` and print cells H12, H19, H26, H35, H42, H50 to confirm they contain formula strings (starting with `=`).
3. Verify no new sheets were added.
4. Verify the formulas reference the correct ranges by printing a sample from each block.

### Critical Constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Do NOT use `data_only=True` when opening (this would strip formulas).
- All formulas must be Excel formula strings, not Python-computed values.
- The lookup formulas MUST use one of the specified patterns (VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH).
- Read the actual file contents carefully before writing any formulas. Adapt to what you find.

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