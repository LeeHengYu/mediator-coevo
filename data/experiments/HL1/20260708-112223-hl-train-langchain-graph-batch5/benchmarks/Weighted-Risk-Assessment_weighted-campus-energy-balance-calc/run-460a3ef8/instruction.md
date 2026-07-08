# Task Instruction

## Task: Weighted Campus Energy Balance Calculation

You must update an Excel workbook with formulas and save the result. Follow these steps precisely.

### Step 0: Inspect the workbook
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Use `openpyxl` to open `/root/output/result.xlsx` and inspect:
   - Sheet `Task`: Read rows 10-50, columns D through L. Print cell values to understand the layout: what's in column D (series codes), what's in row 10 (years), what's in H12:L17, H19:L24, H26:L31 (the yellow cells to fill), H35:L40, H42:L47, H50:L50.
   - Sheet `Data`: Read rows 21-38 to understand the data structure — identify which row/column has series codes, which has years, and how data is laid out.
3. Print all findings before proceeding. Understanding the exact layout is critical.

### Step 1: Populate lookup formulas in yellow cells

For each cell in ranges `H12:L17`, `H19:L24`, and `H26:L31` on sheet `Task`:
- The formula must use TWO inputs: (a) the series code from column D of that row, and (b) the year from row 10 of that column.
- The data source is sheet `Data` rows 21:38.
- Use one of these lookup patterns: `INDEX` with `MATCH`, `VLOOKUP` with `MATCH`, `HLOOKUP` with `MATCH`, or `XLOOKUP` with `MATCH`.
- Use `INDEX/MATCH` as the preferred pattern since it's most flexible. The formula pattern should be something like: `=INDEX(Data!<data_range>, MATCH(<series_code_ref>, Data!<series_code_column>, 0), MATCH(<year_ref>, Data!<year_row>, 0))`
- Make sure references are appropriately absolute (e.g., lock the column for the series code reference with `$D12`, lock the row for the year reference with `H$10`, and use absolute references for the Data ranges).
- IMPORTANT: Inspect the Data sheet carefully to determine:
  - Which column contains the series codes (the lookup key)
  - Which row contains the years (the lookup key)
  - What is the exact data range for the values
  - Adjust the INDEX/MATCH formula accordingly

### Step 2: Net renewable balance and statistics

In `H35:L40`, enter formulas for Net Renewable Balance for each campus:
```
= (Renewable Generation - Grid Consumption) / Baseline Energy Demand * 100
```
where:
- Renewable Generation values are in `H12:L17`
- Grid Consumption values are in `H19:L24`  
- Baseline Energy Demand values are in `H26:L31`
- Match each campus row correctly (row 35 corresponds to the same campus as rows 12, 19, 26; row 36 to rows 13, 20, 27; etc.)

IMPORTANT: Verify which block corresponds to which metric by reading the labels. The mapping above (H12:L17 = Renewable Generation, H19:L24 = Grid Consumption, H26:L31 = Baseline Energy Demand) is assumed — confirm by reading the sheet labels before writing formulas.

In `H42:L47`, enter column-wise statistical formulas over `H35:L40`:
- Row 42: `=MIN(H35:H40)` (adjust column for each)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

IMPORTANT: Verify the expected order of statistics by reading labels in the Task sheet before assigning formulas to rows.

### Step 3: Weighted mean with SUMPRODUCT

In `H50:L50`, calculate the weighted mean for MCEC:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the Net Renewable Balance percentages (H35:H40) as values and the Baseline Energy Demand (H26:H31) as weights. Adjust column references for each column H through L.

### Step 4: Save and verify
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and verify:
   - Cells in H12:L17, H19:L24, H26:L31 contain formulas (not plain values)
   - Cells in H35:L40 contain formulas
   - Cells in H42:L47 contain formulas
   - Cells in H50:L50 contain formulas
   - No new sheets were added
   - Print a sample of formulas to confirm correctness

### Critical constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting (use `openpyxl` with care — when loading, do NOT use `data_only=True` since you need to write formulas).
- Use `keep_vba=False` (default) and be careful not to strip existing formatting. Load with `openpyxl.load_workbook(filename, data_only=False)` to preserve existing formulas if any.
- All formulas must be Excel formulas (strings starting with `=`), not Python-computed values.
- The final file must be at `/root/output/result.xlsx`.

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