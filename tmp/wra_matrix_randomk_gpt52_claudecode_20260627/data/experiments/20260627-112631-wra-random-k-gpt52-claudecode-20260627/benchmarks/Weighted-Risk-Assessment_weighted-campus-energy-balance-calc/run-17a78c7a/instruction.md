# Task Instruction

Execute the following steps precisely to complete the weighted campus energy balance workbook task.

## Step 0: Inspect the workbook
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Open `/root/output/result.xlsx` using openpyxl (with `data_only=False` so formulas are preserved).
3. Print the contents of sheet `Task` rows 1–55, columns A–M, showing both values and any existing formulas. Pay special attention to:
   - Column D rows 12–17, 19–24, 26–31 (series codes)
   - Row 10 columns H–L (years)
   - Row 35–40 labels (campus names)
   - Row 42–47 labels (min, max, median, mean, 25th, 75th percentile)
   - Row 50 label
   - The yellow cell ranges: H12:L17, H19:L24, H26:L31
4. Print sheet `Data` rows 1–40, columns A–Z (or however wide it goes), to understand the data layout, especially rows 21–38. Identify:
   - Which column contains the series codes (lookup keys)
   - Which row contains the year headers
   - The exact column letters and row numbers

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Using openpyxl, write Excel formulas (not computed values) into each cell. For each cell at row `r`, column `c` (H=8, I=9, J=10, K=11, L=12):

- The series code is in `$D{r}` (column D of the same row)
- The year is in the corresponding column of row 10, e.g., `H$10`, `I$10`, etc.
- The data source is on sheet `Data` rows 21:38

Use an INDEX/MATCH/MATCH pattern like:
```
=INDEX(Data!<data_range>, MATCH($D{r}, Data!<series_code_column>, 0), MATCH(<col>$10, Data!<year_header_row>, 0))
```

Adjust the exact ranges based on what you discover in Step 0. The data range, series code column, and year header row must match the actual layout of the Data sheet. Make sure:
- The MATCH for series codes searches the correct column range in Data rows 21:38
- The MATCH for years searches the correct row range in Data (the header row for that data block)
- The INDEX range covers the full data block
- Use absolute references where appropriate ($D for the series code column, $ on row 10 for the year row)

## Step 2a: Net renewable balance formulas in H35:L40

For each campus (rows 35–40), calculate:
```
=(Renewable_Generation - Grid_Consumption) / Baseline_Energy_Demand * 100
```

The Renewable Generation values are in H12:L17, Grid Consumption in H19:L24, and Baseline Energy Demand in H26:L31. Map each campus row:
- Row 35 uses data from rows 12, 19, 26 (first campus)
- Row 36 uses data from rows 13, 20, 27
- Row 37 uses data from rows 14, 21, 28
- Row 38 uses data from rows 15, 22, 29
- Row 39 uses data from rows 16, 23, 30
- Row 40 uses data from rows 17, 24, 31

So for cell H35: `=(H12-H19)/H26*100`
For cell H36: `=(H13-H20)/H27*100`
...and so on for all 6 rows × 5 columns.

## Step 2b: Summary statistics in H42:L47

For each column (H through L), calculate column-wise statistics over the 6 campus values in rows 35:40. Based on the row labels (inspect them in Step 0, but likely):
- Row 42 (Minimum): `=MIN(H35:H40)` 
- Row 43 (Maximum): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

**IMPORTANT**: Verify the actual row labels before assigning formulas. Match each formula to the correct label row.

## Step 3: Weighted mean in H50:L50

For each column (H through L), use SUMPRODUCT with the Net renewable balance percentages (H35:H40) as values and Baseline Energy Demand (H26:H31) as weights:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```

Repeat for columns I, J, K, L.

## Step 4: Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open it and print all formula cells to confirm they are written correctly.
3. Verify no extra sheets were added.
4. Verify formatting was not changed (openpyxl should preserve it if you only write to cell values).
5. Optionally open with xlcalc or print a few cells with data_only=True (re-open) to spot-check computed values make sense.

## Critical Notes
- Write EXCEL FORMULAS as strings starting with `=`, not Python-computed values.
- Do not modify any cells outside the specified ranges.
- Do not add sheets, macros, VBA, or external links.
- Use `openpyxl` to read and write. Load with `data_only=False`.
- When writing formulas, reference the `Data` sheet as `Data!` (match the exact sheet name).
- After inspecting the Data sheet layout, adapt all range references accordingly. The specific column letters and row numbers in the INDEX/MATCH formulas depend on the actual Data sheet structure.

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