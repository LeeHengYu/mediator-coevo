# Task Instruction

Execute the following steps to produce /root/output/result.xlsx.

## 0 – Environment setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1 – Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Print:
- Sheet names
- Sheet `Task`: values/formulas in D12:D17, D19:D24, D26:D31 (series codes), row 10 H10:L10 (years), H35:H40 area labels if any, H42:H47 labels, H50 label, and any existing content in H12:L31.
- Sheet `Data`: rows 21–38 to understand the data layout (columns A through at least M). Print headers row and data rows.

This inspection is critical — do NOT skip it. Record the exact column letters and row numbers for the Data table headers and values.

## 2 – Write a Python script that opens the workbook and populates formulas

Use openpyxl to load the workbook (no data_only), write formulas as strings, and save to `/root/output/result.xlsx`.

### Step 1 formulas (H12:L17, H19:L24, H26:L31)
For each cell at row `r` and column `c` (H=8 … L=12):
- The series code is in column D of the same row on sheet Task, e.g., `D12`.
- The year is in row 10 of the same column, e.g., `H10`.
- The data lives on sheet `Data` rows 21:38.

Determine from the inspection which column of `Data` holds the series codes and which row holds years (likely row 20 or a header row). Then construct an INDEX/MATCH formula like:
```
=INDEX(Data!<data_range>, MATCH(D12, Data!<series_col_range>, 0), MATCH(H10, Data!<year_row_range>, 0))
```
Adjust the exact ranges based on what you see in the inspection. The MATCH for the series code should search down the series-code column of Data rows 21:38. The MATCH for the year should search across the year header row of Data.

Apply this pattern to all 54 cells (18 rows × 5 columns) in the three blocks.

### Step 2 formulas

**H35:L40 – Net production slack** for each of the six plants:
- Row 35 corresponds to the first plant. From the Task sheet layout, the three blocks are: block 1 = rows 12–17 (e.g., Finished Output), block 2 = rows 19–24 (e.g., Scrap And Rework), block 3 = rows 26–31 (Rated Production Capacity).
- Verify which block is which from the labels (column B or C). The formula for row 35, column H is:
  `=(H12-H19)/H26*100`  (adjust row offsets if the blocks map differently)
- Repeat for all 6 plants × 5 years.

**H42:L47 – Column-wise statistics** over H35:L40:
- H42: `=MIN(H35:H40)`
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)`
- H46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- H47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Use the legacy function name `PERCENTILE` — NOT `PERCENTILE.INC` or `PERCENTILE.EXC` or `_xlfn.PERCENTILE.INC`. The prior run failed with #NAME? errors because a non-legacy function name was used. `PERCENTILE` is the universally compatible name.

Also double-check: use `MEDIAN` (not `_xlfn.MEDIAN`), `MIN`, `MAX`, `AVERAGE` — all legacy names.

Repeat for columns I through L.

### Step 3 formula (H50:L50)
Weighted mean for each year column using SUMPRODUCT:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
Wait — re-read the instruction: "using the Step 2 percentages as values and the Rated Production Capacity block in H26:L31 as weights". So the formula is:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
Repeat for I, J, K, L columns.

## 3 – Validate
After saving, re-open `/root/output/result.xlsx` with openpyxl (data_only=False) and print:
- A sample of formulas from each block (H12, L17, H19, L24, H26, L31)
- H35, H40, H42:H47, H50
- Confirm no cell is None or empty where a formula should be
- Confirm PERCENTILE is spelled exactly as `PERCENTILE` (not PERCENTILE.INC)
- Confirm no `_xlfn.` prefix appears anywhere

## 4 – Verify the order of statistics rows
From the inspection, confirm what labels are in column B/C/D for rows 42–47. The order (MIN, MAX, MEDIAN, AVERAGE, 25th pctl, 75th pctl) must match whatever labels are in the sheet. If the labels differ (e.g., row 42 is Maximum, row 43 is Minimum), adjust the formula placement accordingly. Print the labels and your formula assignments to confirm alignment.

## Important constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Save only to `/root/output/result.xlsx`.
- Use `keep_vba=False` (default) when loading.
- When loading, do NOT use `data_only=True` — we need to write formulas, not cached values.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=hard, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.