# Task Instruction

## Task: Update workbook with formulas and save to /root/output/result.xlsx

### Setup
1. `mkdir -p /root/output`
2. Install openpyxl if not already available: `pip install openpyxl`
3. Inspect the workbook structure thoroughly before making any changes.

### Phase 0: Thorough Inspection
Read `/root/data/workbook.xlsx` with openpyxl (with `data_only=False` to see formulas).

**On sheet `Task`:**
- Print all cell values in column D rows 12-31 (series codes for each block).
- Print all cell values in row 10, columns H through L (the years).
- Print cell values in rows 35-40 column D (port names for Net container flow).
- Print cell values in rows 42-47 column D or G (labels for min, max, median, mean, 25th, 75th percentile).
- Print cell values in row 50 columns D-G (CPA weighted mean label area).
- Print any existing content/formulas in H12:L17, H19:L24, H26:L31 to see if they're empty or have values.
- Print the block labels/headers around rows 11, 18, 25, 34, 41, 49 to understand the three data blocks (Loaded Containers Inbound, Loaded Containers Outbound, Terminal Throughput Capacity).
- Print fill colors of cells in the yellow regions to confirm which cells need formulas.

**On sheet `Data`:**
- Print rows 19-40 to understand the data layout (headers, series codes, years).
- Identify which row contains headers and which column contains series codes.
- Identify which row contains year headers and in which row/column orientation the data is arranged.
- Determine if Data is organized with series codes in a column and years in a row, or vice versa.

### Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write a Python script using openpyxl to populate these cells with `INDEX(MATCH,MATCH)` formulas.

The formula pattern for each cell should be:
```
=INDEX(Data!<data_range>, MATCH(<series_code_ref>, Data!<series_code_column>, 0), MATCH(<year_ref>, Data!<year_row>, 0))
```

Where:
- `<series_code_ref>` = reference to column D of the current row on Task sheet (e.g., `$D12`)
- `<year_ref>` = reference to the year in row 10 of the current column (e.g., `H$10`)
- `<data_range>`, `<series_code_column>`, `<year_row>` are determined from the Data sheet layout in rows 21:38.

**IMPORTANT**: The instruction says source records are in Data rows 21:38. Inspect carefully to determine:
- Which column in Data contains the series codes
- Which row in Data contains the year headers
- What the actual data range is

Use `$D{row}` (column-absolute) for series code and `{col}$10` (row-absolute) for year to allow proper copying behavior.

Make sure the INDEX range, MATCH lookup ranges all reference the Data sheet correctly.

### Phase 2: Net container flow in H35:L40

For each cell in H35:L40, the formula should be:
```
=(H12 - H19) / H26 * 100
```
Adjusted for the correct rows. Specifically:
- Row 35 uses data from rows 12, 19, 26 (first port)
- Row 36 uses data from rows 13, 20, 27 (second port)
- Row 37 uses data from rows 14, 21, 28
- Row 38 uses data from rows 15, 22, 29
- Row 39 uses data from rows 16, 23, 30
- Row 40 uses data from rows 17, 24, 31

Verify this mapping by checking that the port names/order in rows 35-40 match those in rows 12-17.

### Phase 3: Summary statistics in H42:L47

For each column H through L:
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

**IMPORTANT**: Check the actual labels in column D/G for rows 42-47 to confirm the order (min, max, median, mean, 25th, 75th). Adjust row assignments accordingly.

### Phase 4: Weighted mean in H50:L50

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

This uses the Net container flow percentages as values and Terminal Throughput Capacity as weights.

### Phase 5: Save and Validate
1. Save to `/root/output/result.xlsx`
2. Re-open the saved file and verify:
   - Formulas exist in all target cells (H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50)
   - No extra sheets were added
   - Print a sample of formulas from each block to confirm correctness
   - Verify the workbook has exactly the original sheets (`Task` and `Data`)

### Critical Constraints
- Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting (do not modify fonts, fills, borders, etc.).
- Use openpyxl and be careful to preserve existing content and formatting.
- When opening the workbook, do NOT use `data_only=True` (that would strip formulas).
- All formulas must be Excel formulas stored as strings starting with `=`.
- The lookup formulas must use one of: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH. Prefer INDEX+MATCH.

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