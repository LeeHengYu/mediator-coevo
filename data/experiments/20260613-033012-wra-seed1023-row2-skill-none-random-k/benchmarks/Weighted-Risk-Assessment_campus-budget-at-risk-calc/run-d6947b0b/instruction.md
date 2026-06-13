# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Preparation

```bash
mkdir -p /root/output
pip install openpyxl
```

Then open a Python session and keep it open throughout.

## 1. Inspect the workbook structure

Read `/root/data/workbook.xlsx` with openpyxl (data_only=False so we see formulas). Inspect:

- Sheet names (should be `Task` and `Data`).
- On sheet `Task`:
  - Row 10: read cells H10:L10 to find the years.
  - Column D rows 12-17, 19-24, 26-31: read the series codes.
  - Rows 35-40 column D: department names or series codes for Net budget buffer.
  - Row 42-47 column A or B or C: labels for min, max, median, mean, 25th, 75th percentile.
  - Row 50: label for Campus Budget Council weighted mean.
  - Cells H12:L17, H19:L24, H26:L31: confirm they are empty (yellow cells to fill).
  - Cells H35:L40, H42:L47, H50:L50: confirm empty.
- On sheet `Data`:
  - Rows 21:38: inspect the layout. Identify which column holds series codes, and which columns/rows hold year headers and values. Print rows 19-38 and columns A-Z (or however wide the data goes) to understand the lookup structure.

Print all of this information before proceeding. This is critical for writing correct formulas.

## 2. Understand the data layout on sheet `Data`

From the inspection, determine:
- Which column on `Data` contains the series codes (likely column A or B).
- Which row on `Data` contains year headers.
- The exact range of data rows 21:38.
- Whether the lookup should be horizontal or vertical.

This determines the correct MATCH/INDEX or VLOOKUP pattern.

## 3. Write formulas for Step 1 (H12:L31)

For each cell in H12:L17, H19:L24, H26:L31, write an Excel formula using INDEX/MATCH (or VLOOKUP with MATCH, etc.) that:
- Takes the series code from column D of the same row on `Task`.
- Takes the year from row 10 of the same column on `Task`.
- Looks up the value from `Data` rows 21:38.

Use absolute references for the Data range and MATCH ranges. Use relative references for the series code (column D, same row) and year (row 10, same column).

Example pattern (adjust based on actual data layout discovered in step 1):
- If Data has series codes in column A and years across columns as headers in some row:
  `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
- Adjust column/row references based on actual inspection.

Write these formulas using openpyxl by assigning formula strings to cells. Use `ws['H12'] = '=INDEX(...)'` syntax.

## 4. Write formulas for Step 2 (H35:L40 and H42:L47)

For H35:L40 — Net budget buffer:
- The formula is: `(Committed Funding - Operating Spend) / Approved Budget Base * 100`
- From the Task sheet layout:
  - H12:L17 likely corresponds to one block (e.g., Committed Funding)
  - H19:L24 likely corresponds to another block (e.g., Operating Spend)
  - H26:L31 likely corresponds to Approved Budget Base
- Verify which block is which by reading labels near rows 11, 18, 25 on the Task sheet.
- For each cell in H35:L40, write the formula, e.g.: `=(H12-H19)/H26*100` (adjusting row references to match the correct department row within each block).

For H42:L47 — Summary statistics (column-wise over H35:L40):
- Read the labels in column A/B/C for rows 42-47 to determine which statistic goes where.
- Use these Excel functions:
  - MIN: `=MIN(H35:H40)`
  - MAX: `=MAX(H35:H40)`
  - MEDIAN: `=MEDIAN(H35:H40)`
  - AVERAGE: `=AVERAGE(H35:H40)`
  - 25th percentile: `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
  - 75th percentile: `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`
- Match each formula to the correct row based on the label.

## 5. Write formulas for Step 3 (H50:L50)

Weighted mean using SUMPRODUCT:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted average of Net budget buffer percentages weighted by Approved Budget Base. Write this for each column H through L.

## 6. Save the workbook

Save to `/root/output/result.xlsx` using openpyxl's `wb.save()`. Do NOT use `data_only=True` when loading — preserve formulas.

## 7. Verify

Reload the saved file and verify:
- Cells H12:L17, H19:L24, H26:L31 contain formula strings (not None or values).
- Cells H35:L40 contain formula strings.
- Cells H42:L47 contain formula strings.
- Cells H50:L50 contain formula strings.
- No new sheets were added.
- Print a sample of formulas to confirm correctness.

## IMPORTANT NOTES
- Do NOT use data_only=True when opening the workbook for editing.
- Keep all existing formatting — do not clear or overwrite cell styles.
- Only write to the specified yellow cells; do not modify any other cells.
- All formulas must be Excel formulas (strings starting with '='), not computed Python values.
- Use the exact block references discovered during inspection. Do not guess — inspect first, then write formulas.

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