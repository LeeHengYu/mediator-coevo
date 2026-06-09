# Task Instruction

Execute the following steps to produce /root/output/result.xlsx.

## Phase 0 – Setup
```python
import openpyxl, os, shutil
os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')
wb = openpyxl.load_workbook('/root/output/result.xlsx')   # do NOT use data_only=True
ts = wb['Task']
ds = wb['Data']
```

## Phase 1 – Inspect the workbook structure
Before writing any formulas, print the following so you can verify coordinates:
1. **Task sheet row 10** (the year header row): print cells A10 through L10 values.
2. **Task sheet column D** for rows 12–31: print each cell's value (these are the series codes used for lookups).
3. **Task sheet rows 35–40 column D**: print each cell's value (port names or labels for Net container flow).
4. **Task sheet rows 42–47 column C or D**: print labels (min, max, median, mean, 25th, 75th percentile labels).
5. **Task sheet row 50 columns C–D**: print the CPA label row.
6. **Data sheet row 20 or 21**: print columns A–Z to find the header row of the data table (series codes in one column, years across columns).
7. **Data sheet column A (or whichever column holds series codes) rows 21–38**: print values.
8. **Data sheet row that holds years** (likely row 20 or the first row of the 21:38 block): print values across columns to identify where years appear.

Use this inspection to determine:
- `DATA_CODE_COL`: which column in Data holds the series codes (e.g., column A or B).
- `DATA_YEAR_ROW`: which row in Data holds the year headers.
- `DATA_FIRST_COL` / `DATA_LAST_COL`: the column range holding numeric data.
- The exact row range for the data body.

## Phase 2 – Write lookup formulas (H12:L17, H19:L24, H26:L31)
Use `INDEX/MATCH` with the structure:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<code_column_range>, 0), MATCH(H$10, Data!<year_row_range>, 0))
```
Where:
- `$D12` uses a column-absolute reference so it stays fixed when filling across columns.
- `H$10` uses a row-absolute reference so it stays fixed when filling down rows.
- The `Data!<data_range>` should cover the numeric body of rows 21:38 (excluding headers).
- The `Data!<code_column_range>` is the single column of series codes in Data rows 21:38.
- The `Data!<year_row_range>` is the single row of year headers in Data.

Loop over the three blocks (rows 12–17, 19–24, 26–31) and columns H–L (columns 8–12) to write the formula string into each cell. Use `f-strings` to build the formula. Make sure the Data sheet references use the correct absolute/relative anchoring.

**Critical**: Construct the formula referencing the actual discovered coordinates from Phase 1. Do not hardcode Data sheet coordinates without verifying them first.

## Phase 3 – Net container flow (H35:L40)
The formula for each cell is:
```
=(H12 - H19) / H26 * 100
```
where H12 is Loaded Containers Inbound, H19 is Loaded Containers Outbound, H26 is Terminal Throughput Capacity, adjusted for the correct row offsets for each of the 6 ports.

For row 35 (first port): `=(<col>12 - <col>19) / <col>26 * 100`
For row 36 (second port): `=(<col>13 - <col>20) / <col>27 * 100`
...and so on through row 40.

Use cell references (e.g., `H12`, `H19`, `H26`) not named ranges.

## Phase 4 – Statistics (H42:L47)
For each column (H through L):
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (AVERAGE): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**Important**: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) unless inspection of existing formulas in the workbook suggests otherwise. The previous cross-task feedback indicates `PERCENTILE` worked for the port throughput variant.

## Phase 5 – Weighted mean for CPA (H50:L50)
For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the Net container flow percentages as values and Terminal Throughput Capacity as weights.

## Phase 6 – Save and verify
```python
wb.save('/root/output/result.xlsx')
```
Then reload the workbook (without data_only) and print cells from each section to confirm formulas are present (not None):
- Print H12, I12, L17 (lookup block)
- Print H35, L40 (net flow block)
- Print H42, H47 (stats block)
- Print H50, L50 (weighted mean)

All printed values should be formula strings starting with `=`. If any are None, debug immediately.

## Constraints
- Do NOT use `data_only=True` when loading.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify formatting.
- Do NOT overwrite non-yellow cells.

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