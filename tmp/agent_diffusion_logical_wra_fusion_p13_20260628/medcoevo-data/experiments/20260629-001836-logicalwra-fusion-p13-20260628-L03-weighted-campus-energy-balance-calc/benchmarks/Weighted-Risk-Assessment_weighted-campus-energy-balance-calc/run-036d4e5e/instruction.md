# Task Instruction

## Task: Populate formulas and cached values in `/root/data/workbook.xlsx`, save to `/root/output/result.xlsx`

### Phase 0: Setup and Inspection
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: read row 10 (years in H10:L10), column D rows 12-17, 19-24, 26-31 (series codes), rows 35-40 labels, row 42-47 labels, row 50 label.
   - Sheet `Data`: read rows 21-38 to understand the data layout. Identify which column holds the series codes (likely column D) and which columns hold year data (likely H onward). Note the exact row range and column range.
   - Record all these values precisely before writing any formulas.

### Phase 1: Write Lookup Formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write an `INDEX/MATCH` formula that:
- Uses the series code from column D of the same row on `Task`
- Uses the year from row 10 of the same column on `Task`
- Looks up in `Data` sheet rows 21:38

The formula pattern for cell `H12` (adjust references for each cell):
```
=INDEX(Data!$H$21:$L$38, MATCH($D12, Data!$D$21:$D$38, 0), MATCH(H$10, Data!$H$20:$L$20, 0))
```
**IMPORTANT**: Before writing this formula, verify:
- What row on `Data` contains the year headers that match Task row 10. It might be row 20 or another row. Inspect `Data` to find the row with years matching H10:L10 on Task.
- That `Data!$D$21:$D$38` contains the series codes matching Task column D.
- Adjust the INDEX range and MATCH ranges accordingly.

Fill all 18 cells in each of the three blocks (6 rows × 5 columns each = 90 cells total across all three blocks) using the same pattern with appropriate absolute/relative references.

### Phase 2: Net Renewable Balance in H35:L40
For each of the 6 campuses (rows 35-40) and 5 years (columns H-L):
```
= (H12 - H19) / H26 * 100
```
Where:
- H12:L17 = Renewable Generation block (rows 12-17 map to rows 35-40 by campus)
- H19:L24 = Grid Consumption block (rows 19-24)
- H26:L31 = Baseline Energy Demand block (rows 26-31)

So for H35: `=(H12-H19)/H26*100`, for H36: `=(H13-H20)/H27*100`, etc.

### Phase 3: Summary Statistics in H42:L47
For each column H through L:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

**IMPORTANT**: Check the row labels in column B/C/D of rows 42-47 to confirm which statistic goes in which row. The order above is a guess — match the actual labels.

### Phase 4: Weighted Mean in H50:L50
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of net renewable balance using Baseline Energy Demand as weights.

### Phase 5: Inject Cached Values
The verifier likely reads cached values (not live formula results). After writing all formulas:
1. Use Python with openpyxl to compute the numeric result of each formula cell.
2. For each formula cell, set the `value` attribute in the underlying XML `<v>` element so that openpyxl's `data_only=True` mode and any CSV export will see the computed numbers.

Concrete approach:
- After writing all formulas with openpyxl, save the workbook.
- Re-open it, read the Data sheet values into a dictionary keyed by (series_code, year).
- Compute each lookup result numerically in Python.
- Compute each net renewable balance, statistics, and weighted mean numerically.
- Re-open the saved file with openpyxl, and for each formula cell, set `cell.value` to the formula string AND manually set the cached value by writing to `cell._value` or by using the internal `cell.value` approach. 
- Actually, the cleanest way: use openpyxl to write formulas, save, then use `zipfile` + `lxml` to directly edit the `xl/worksheets/sheet1.xml` (Task sheet) to add `<v>numeric_value</v>` inside each `<c>` element that has a formula `<f>`. This ensures both the formula and cached value are present.

### Phase 6: Validation
1. Re-open `/root/output/result.xlsx` with `data_only=True` and verify that formula cells return numeric values (not None).
2. Spot-check a few lookup values against the Data sheet.
3. Verify no extra sheets were added.
4. Verify formatting is preserved (spot-check fill colors on a few cells).

### Key Pitfalls to Avoid
- Do NOT assume row/column positions — inspect the actual workbook first.
- The year header row on Data might not be row 20; find it by inspection.
- The statistics order (min/max/median/mean/percentiles) must match the actual row labels.
- Cached `<v>` values must be present for the verifier to read numeric results.
- Do not add sheets, macros, VBA, or external links.
- Preserve all existing formatting.

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