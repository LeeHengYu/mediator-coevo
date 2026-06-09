# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
- Open `/root/data/workbook.xlsx` with openpyxl (data_only=False to preserve formulas).
- Print the sheet names to confirm `Task` and `Data` exist.
- Print the contents of `Task` sheet rows 1–55, columns A–M, to understand the layout: what's in column D (series codes), what's in row 10 (years), what the yellow cell ranges look like, and what labels exist for rows 35–50.
- Print the contents of `Data` sheet rows 18–40, all columns, to understand the lookup source data structure (headers, series codes, years, values).
- Pay special attention to:
  - The exact series codes in column D of `Task` rows 12–17, 19–24, 26–31.
  - The exact years in row 10 of columns H–L on `Task`.
  - The layout of `Data` rows 21–38: which row has headers, which column has series codes, which columns/rows have years and values.
  - The labels in rows 35–40 (department names or series references for Net budget buffer).
  - The labels in rows 42–47 (min, max, median, mean, 25th, 75th percentile).
  - Row 50 label for Campus Budget Council weighted mean.

## 2. Determine the lookup structure
Based on the inspection:
- Identify whether Data rows 21–38 are organized with series codes in a column and years across columns (suitable for VLOOKUP+MATCH or INDEX+MATCH), or series codes in a row and years down rows (suitable for HLOOKUP+MATCH).
- Choose INDEX+MATCH as the lookup pattern (most flexible).
- Determine the exact cell references for the Data lookup range: the column containing series codes, the row containing years, and the data area.

## 3. Populate H12:L17, H19:L24, H26:L31 with lookup formulas
Using openpyxl, write formulas into each cell. For each cell at row `r`, column `c` (H=8, I=9, J=10, K=11, L=12):
- The series code is in `Task!D{r}` (e.g., `$D12`, `$D13`, etc.).
- The year is in `Task!{col}10` (e.g., `H$10`, `I$10`, etc.).
- Use INDEX+MATCH pattern referencing the Data sheet. The exact formula depends on the Data layout discovered in step 1.
- Example formula pattern (adjust based on actual Data layout): `=INDEX(Data!$B$22:$Z$38,MATCH($D12,Data!$A$22:$A$38,0),MATCH(H$10,Data!$B$21:$Z$21,0))`
- Use appropriate absolute/relative references: lock the series code column with `$D` and lock the year row with `$10`, lock the Data ranges absolutely.
- Apply the same formula pattern to all three blocks (H12:L17, H19:L24, H26:L31), since each block's rows have their own series codes in column D.

## 4. Populate H35:L40 with Net budget buffer formulas
The formula is: `(Committed Funding - Operating Spend) / Approved Budget Base * 100`
- From the inspection, identify which of the three blocks (rows 12-17, 19-24, 26-31) corresponds to Committed Funding, Operating Spend, and Approved Budget Base. The block labels should be visible in the Task sheet.
- For each department row `r` in 35–40 and column `c` in H–L, write a formula like:
  `=({CommittedFunding_cell} - {OperatingSpend_cell}) / {ApprovedBudgetBase_cell} * 100`
  where the cells reference the corresponding row in each block and the same column.
- The six departments in rows 35–40 should correspond to the six rows within each block (rows 1–6 of each block map to rows 35–40).

## 5. Populate H42:L47 with summary statistics
For each column `c` in H–L:
- H42 (or whichever row is minimum): `=MIN(H35:H40)` (adjust column letter)
- Maximum: `=MAX(H35:H40)`
- Median: `=MEDIAN(H35:H40)`
- Mean: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)`
- Match each function to the correct row based on the labels found in column A/B/C/D of rows 42–47.

## 6. Populate H50:L50 with weighted mean
For each column `c` in H–L:
- `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)` (adjust column letters)
- This uses the Net budget buffer percentages (H35:H40) as values and Approved Budget Base (H26:H31) as weights.

## 7. Save and validate
- Save the workbook to `/root/output/result.xlsx`.
- Reopen the saved file and print all formula cells to verify they are correctly written.
- Verify no new sheets were added, no macros, no external links.
- Spot-check a few formulas to ensure correct cell references.

## IMPORTANT NOTES
- Do NOT use data_only=True when loading; you must preserve existing formulas and formatting.
- Do NOT modify any cells outside the specified ranges.
- Do NOT change formatting, add sheets, or add macros.
- When writing formulas with openpyxl, prefix them with `=`.
- All Data sheet references in formulas must use the sheet name prefix `Data!`.
- Double-check every formula references the correct rows and columns by cross-referencing with the actual workbook layout discovered in step 1.
- If the Data sheet layout differs from assumptions, adapt the INDEX+MATCH formula accordingly before writing any cells.

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