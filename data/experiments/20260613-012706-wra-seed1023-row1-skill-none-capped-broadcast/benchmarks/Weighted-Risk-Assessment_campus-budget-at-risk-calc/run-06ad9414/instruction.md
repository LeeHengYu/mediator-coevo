# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Inspect the workbook

1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Open `/root/output/result.xlsx` with openpyxl (with `data_only=False` so formulas are preserved).
3. Print out the `Task` sheet structure:
   - Print rows 1–55, columns A–M, showing both values and any existing formulas.
   - Pay special attention to:
     - Column D rows 12–17, 19–24, 26–31 (series codes)
     - Row 10 columns H–L (years)
     - Rows 35–40 (department names or references)
     - Rows 42–47 (stat labels)
     - Row 50 (Campus Budget Council)
4. Print out the `Data` sheet structure:
   - Print rows 1–40, all populated columns.
   - Identify the layout: which column has series codes, which row has years, and where the data values are (rows 21–38).
5. Based on inspection, identify:
   - The exact column letter on `Data` that contains series codes (for MATCH lookup)
   - The exact row on `Data` that contains years (for MATCH lookup)
   - The data range on `Data` (rows 21:38)

## 1. Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each of the three blocks (rows 12–17, 19–24, 26–31), and for each cell in columns H through L:
- The formula must look up the value from the `Data` sheet using:
  - The series code from column D of the current row on `Task`
  - The year from row 10 of the current column on `Task`
- Use an INDEX/MATCH pattern (or VLOOKUP/MATCH, HLOOKUP/MATCH, XLOOKUP/MATCH — pick whichever fits the Data layout best).
- Example pattern if Data has series codes in column A and years in row 1, with data in a rectangular block:
  `=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$1:$XX$1, 0))`
  Adjust column/row references based on actual inspection.
- Make sure the series-code reference uses an absolute column (`$D12`) and the year reference uses an absolute row (`H$10`) so formulas copy correctly across the block.
- Write these formulas as strings into the cells using openpyxl (e.g., `ws['H12'] = '=INDEX(...)'`).

## 2. Net budget buffer in H35:L40

Based on the sheet layout, the three blocks likely correspond to:
- Rows 12–17: Committed Funding
- Rows 19–24: Operating Spend  
- Rows 26–31: Approved Budget Base

(Verify this by reading row labels/headers near rows 11, 18, 25 on the Task sheet.)

For each cell in H35:L40 (6 departments × 5 years):
`= (CommittedFunding - OperatingSpend) / ApprovedBudgetBase * 100`

The row offset within each block should match: row 35 corresponds to the first department (row 12, 19, 26), row 36 to the second (row 13, 20, 27), etc.

Example for H35: `=(H12-H19)/H26*100`
Example for H36: `=(H13-H20)/H27*100`
...and so on through H40: `=(H17-H24)/H31*100`

Apply the same pattern for columns I, J, K, L.

## 3. Summary statistics in H42:L47

For each column (H through L), compute column-wise stats over the 6 values in rows 35–40:
- Row 42 (Minimum): `=MIN(H35:H40)`
- Row 43 (Maximum): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

**Important**: Verify the actual labels in column A/B/C/D for rows 42–47 to confirm which row is which statistic. Assign formulas accordingly — do NOT assume the order above; match the label.

## 4. Weighted mean in H50:L50

For each column (H through L):
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net budget buffer percentages (H35:H40) weighted by Approved Budget Base (H26:H31).

## 5. Save and validate

1. Save the workbook to `/root/output/result.xlsx`.
2. Reopen it and print rows 10–50, columns D–L to verify all formula cells are populated.
3. Confirm:
   - No new sheets were added
   - No macros or VBA
   - All yellow-cell ranges (H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50) contain formulas
   - The formulas reference correct ranges
4. Optionally open with xlcalc or load with data_only=True after a re-save to spot-check computed values make sense (positive or negative percentages, stats in reasonable ranges).

## Key cautions
- Read the actual file carefully before writing any formulas. The exact column letters and row numbers on the Data sheet, and the exact labels on the Task sheet, must be verified by inspection.
- Do not add any sheets, helper columns, macros, or external links.
- Preserve all existing formatting — only write formula values into the specified cells.
- Use `$` signs appropriately in formulas for mixed references so they work across the rectangular blocks.
- If row labels for stats (rows 42–47) differ from the order I assumed, adjust accordingly.

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