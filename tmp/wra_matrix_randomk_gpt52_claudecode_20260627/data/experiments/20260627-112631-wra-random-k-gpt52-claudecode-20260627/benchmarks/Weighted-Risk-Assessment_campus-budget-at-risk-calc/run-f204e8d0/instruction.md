# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl (with `data_only=False` so formulas are preserved). Inspect:
- Sheet `Task`: Print the contents of rows 10-50, columns D through L (especially D10:L10 for years, D12:D17 for series codes in block 1, D19:D24 for block 2, D26:D31 for block 3, D35:D40 for department names or references). Print cell fill colors for H12:L17 to confirm the yellow target cells.
- Sheet `Data`: Print rows 21-38 fully (all columns) to understand the data layout — identify which row contains headers, which column contains series codes, and how years map to columns.

Print everything clearly before making any edits.

## 2. Understand the data layout on sheet `Data`
From the inspection, determine:
- Where series codes appear (likely column A or B of rows 21-38)
- Where year headers appear (likely in a header row, maybe row 21 or row 20)
- The exact column range for data values

This is critical for building correct MATCH/lookup formulas.

## 3. Populate H12:L17, H19:L24, H26:L31 with lookup formulas (Step 1)

For each cell in these ranges, write a formula that:
- Uses the series code from column D of that row
- Uses the year from row 10 of that column (e.g., H10, I10, etc.)
- Looks up the value from `Data!` rows 21:38

Use INDEX/MATCH pattern (most reliable):
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Adjust the exact ranges based on what you found in the inspection. The `$D12` reference should lock the column (D) but not the row; `H$10` should lock the row (10) but not the column. This allows the formula to be written for each cell correctly.

Write formulas cell by cell or use a Python loop with openpyxl to set each cell's value to the appropriate formula string. Make sure:
- Row references in column D change per row
- Column references in row 10 change per column (H through L)
- The Data sheet range references are absolute where needed

## 4. Populate H35:L40 with Net Budget Buffer formulas (Step 2)

From the task structure:
- Block H12:L17 = one metric (likely Committed Funding, or Operating Spend, or Approved Budget Base)
- Block H19:L24 = another metric
- Block H26:L31 = another metric

Identify which block corresponds to which metric by checking the series codes or labels. The formula is:
```
(Committed Funding - Operating Spend) / Approved Budget Base * 100
```

For each cell in H35:L40, write a formula referencing the corresponding cells in the three blocks above. For example, if row 35 corresponds to the first department and H12=Committed Funding, H19=Operating Spend, H26=Approved Budget Base:
```
=(H12-H19)/H26*100
```

Adjust row mappings based on actual department ordering. Verify that the six departments in rows 35-40 match the order in rows 12-17 (and 19-24, 26-31).

## 5. Populate H42:L47 with summary statistics (Step 2 continued)

For each column H through L:
- H42 (MIN): `=MIN(H35:H40)`
- H43 (MAX): `=MAX(H35:H40)`
- H44 (MEDIAN): `=MEDIAN(H35:H40)`
- H45 (MEAN): `=AVERAGE(H35:H40)`
- H46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- H47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`

Check the labels in column D or nearby to confirm which row is which statistic, and adjust the row assignments accordingly.

## 6. Populate H50:L50 with weighted mean (Step 3)

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

This computes the weighted mean of the Net Budget Buffer percentages using Approved Budget Base as weights.

## 7. Preserve formatting
- Open the workbook without `data_only` so formulas are kept.
- Do NOT modify any cell formatting, styles, fills, fonts, borders, or number formats.
- Do NOT add or remove sheets.
- Only set `.value` on the target cells.

## 8. Save
Save the workbook to `/root/output/result.xlsx`.

## 9. Verify
Reopen `/root/output/result.xlsx` and print:
- All formula cells you wrote (H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50) to confirm they contain formula strings (starting with `=`)
- Confirm no extra sheets were added
- Confirm the formulas reference correct ranges

## IMPORTANT NOTES
- You MUST inspect the actual workbook structure before writing any formulas. The exact row/column layout on the Data sheet determines the formula ranges.
- Use `data_only=False` when loading so existing formulas are preserved.
- All formulas must be Excel formula strings (starting with `=`), not computed Python values.
- The labels in rows 42-47 might be in a different order than MIN/MAX/MEDIAN/MEAN/P25/P75 — read the actual labels to determine correct placement.
- For PERCENTILE, check if the workbook uses PERCENTILE, PERCENTILE.INC, or PERCENTILE.EXC by looking at any existing formulas or context clues. Default to PERCENTILE.INC if unsure.

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