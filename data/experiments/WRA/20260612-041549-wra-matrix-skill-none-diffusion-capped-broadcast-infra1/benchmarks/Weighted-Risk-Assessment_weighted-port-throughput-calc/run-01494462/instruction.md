# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Phase 0 – Inspect the workbook structure
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl and print:
   - Sheet names.
   - From sheet `Task`: cells D12:D17 (series codes block 1), D19:D24 (block 2), D26:D31 (block 3), row 10 columns H–L (years), cells H35:H40 labels or D35:D40 if port names are there, and cells H42:H47 / D42:D47 (stat labels), and D50 or label near H50.
   - From sheet `Data`: rows 21–38, all populated columns, to understand the data layout (which column holds the series code, which row/column holds years, and where numeric values sit).
   Print everything clearly so you can design formulas.

## Phase 1 – Populate lookup blocks H12:L17, H19:L24, H26:L31
Using openpyxl, write **string formulas** (not computed values) into every yellow cell in those three blocks. Each formula must use the `INDEX/MATCH` pattern:
```
=INDEX(Data!<data_range>, MATCH(<series_code_cell>, Data!<series_code_column>, 0), MATCH(<year_cell>, Data!<year_row>, 0))
```
where:
- `<series_code_cell>` = the cell in column D of the current row on sheet Task (e.g. $D12).
- `<year_cell>` = the cell in row 10 for the current column on sheet Task (e.g. H$10).
- `<data_range>`, `<series_code_column>`, and `<year_row>` are determined from Phase 0 inspection of sheet Data rows 21–38.

Lock references appropriately with `$` so formulas can be conceptually dragged across the 5×6 grid but write each cell individually.

## Phase 2 – Net container flow (H35:L40)
For each of the 6 ports and 5 years, write a formula:
```
=(H12 - H19) / H26 * 100
```
adjusted so that row 12→35, 13→36, …, 17→40 correspond to the same port, and columns H–L correspond to the same year. Use relative references to the cells populated in Phase 1 (Loaded Containers Inbound = rows 12–17, Loaded Containers Outbound = rows 19–24, Terminal Throughput Capacity = rows 26–31).

## Phase 3 – Summary statistics (H42:L47)
For each column H–L, write these formulas in the six rows 42–47:
- Row 42 (minimum): `=MIN(H35:H40)`
- Row 43 (maximum): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

**Important:** Verify the exact label text in cells to the left of rows 42–47 during Phase 0 to confirm which row maps to which statistic. Adjust the row mapping if the labels differ from the assumed order above.

## Phase 4 – Weighted mean (H50:L50)
For each column H–L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net container flow percentages using Terminal Throughput Capacity as weights.

## Phase 5 – Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open `/root/output/result.xlsx` and print:
   - A sample of cells from each block (e.g. H12, L17, H19, L24, H26, L31) to confirm they contain formula strings (not None).
   - Cells H35, L40, H42, L47, H50, L50 to confirm formulas are present.
3. If any cell is None or empty, investigate and fix before finishing.

## Critical constraints
- Use openpyxl; do NOT use data_only mode when writing.
- Write formulas as strings (e.g. `ws['H12'] = '=INDEX(...)'`), never computed Python values.
- Do NOT add or remove sheets, do NOT add macros/VBA/external links.
- Do NOT alter any existing formatting, merged cells, or non-target cells.
- Use `PERCENTILE.INC` if unsure (it's the Excel default for PERCENTILE).
- Adapt all ranges and references based on what you actually observe in Phase 0. The row/column numbers above are from the task description but confirm them against the actual file.

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