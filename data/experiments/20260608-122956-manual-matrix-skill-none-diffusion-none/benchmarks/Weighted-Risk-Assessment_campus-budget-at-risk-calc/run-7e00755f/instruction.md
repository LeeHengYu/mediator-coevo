# Task Instruction

Execute the following steps to complete the task.

## Step 0 – Inspect the workbook
1. Copy the source workbook:
   ```bash
   mkdir -p /root/output
   cp /root/data/workbook.xlsx /root/output/result.xlsx
   ```
2. Open `/root/output/result.xlsx` with openpyxl (do NOT use `data_only=True`) and inspect:
   - **Task sheet**: Print the contents of rows 10-50, columns D through L. Pay special attention to:
     - Row 10 (year headers in H10:L10)
     - Column D rows 12-17, 19-24, 26-31 (series codes)
     - Row labels in column B or C for rows 12-17 (should correspond to department names or data categories like Committed Funding, Operating Spend, Approved Budget Base)
     - Rows 35-40 labels and row 42-47 labels (MIN, MAX, MEDIAN, AVERAGE/MEAN, 25th percentile, 75th percentile)
     - Row 50 label
   - **Data sheet**: Print rows 19-40 to understand the layout. Identify:
     - Which column contains the series codes (likely column A or B)
     - Which row contains year headers
     - The data range that rows 21:38 span
   Print all of this clearly so we can construct correct formulas.

## Step 1 – Write lookup formulas in H12:L31
Using the inspection results, write INDEX/MATCH formulas into the yellow cells H12:L17, H19:L24, and H26:L31.

The general pattern for each cell should be:
```
=INDEX(Data!<data_range>, MATCH($D<row>, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Critical details:
- `$D<row>` – use column-absolute reference so the series code is locked when filling across columns.
- `H$10` – use row-absolute reference so the year header row is locked when filling down.
- `<data_range>` – the rectangular block on the Data sheet containing the numeric values (rows 21:38, but determine the exact column span from inspection).
- `<series_code_column>` – the column on Data that holds the series codes, same rows as data_range.
- `<year_header_row>` – the row on Data that holds the year values, same columns as data_range.
- Make sure the MATCH ranges align exactly with the INDEX array dimensions.

Write these formulas using openpyxl by iterating over each cell in the three blocks.

## Step 2 – Net budget buffer formulas in H35:L40
For each cell in H35:L40, write a formula that computes:
```
=(H12 - H19) / H26 * 100
```
where the row offsets correspond to:
- Committed Funding block: rows 12-17 (H12:L17)
- Operating Spend block: rows 19-24 (H19:L24)
- Approved Budget Base block: rows 26-31 (H26:L31)

So for row 35 col H: `=(H12-H19)/H26*100`, row 36 col H: `=(H13-H20)/H27*100`, etc.
Adjust if the inspection reveals different row mappings between the department rows in each block and the department rows in H35:L40. The key is that department order matches.

## Step 3 – Statistical summary in H42:L47
For each column H through L, write:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

Verify the row labels from inspection match this order. If labels differ (e.g., row 42 is MAX not MIN), adjust accordingly.

## Step 4 – Weighted mean in H50:L50
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net budget buffer percentages using Approved Budget Base as weights.

## Step 5 – Save and verify
1. Save the workbook with `wb.save('/root/output/result.xlsx')`. Do NOT use `data_only=True` at any point.
2. Re-open the saved file and print the formula strings in a sample of cells (e.g., H12, L17, H35, H42, H50) to confirm they are present and correctly structured.
3. Confirm no new sheets were added.

## Important constraints
- Do NOT use `data_only=True` when loading – this would strip formulas.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Preserve all existing formatting.
- Use openpyxl throughout.

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