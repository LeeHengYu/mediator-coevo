# Task Instruction

Execute the following steps in a single Python script to produce both deliverables.

## Step 0 – Inspect the source
```python
import openpyxl
wb_src = openpyxl.load_workbook('/root/Promo_Price_Check_Source.xlsx')
for name in wb_src.sheetnames:
 ws = wb_src[name]
 print(f'Sheet: {name}, rows={ws.max_row}, cols={ws.max_column}')
 for r in range(1, min(ws.max_row+1, 5)):
 print([ws.cell(r, c).value for c in range(1, ws.max_column+1)])
```
Run this first to see the exact column headers, data types (especially dates vs strings), and row count. Use the output to map source columns to the 8 required columns.

## Step 1 – Build `/root/Promo_Register_Audit.xlsx`

Use `openpyxl` throughout (do NOT use pandas to avoid any date-parsing surprises).

### 1a) `RawData` sheet
- Copy every cell from the source workbook's first (or only) sheet into a new workbook's sheet named `RawData`, preserving values exactly (including dates as datetime objects).

### 1b) `Formatted Data` sheet
- Create sheet named exactly `Formatted Data`.
- Row 1 headers (columns A-L): `Promo ID`, `SKU`, `Promo Price`, `Register Price`, `Promo Start Date`, `Sale Date`, `Promo End Date`, `Store ID`, `Price Error`, `Window Error`, `Total Errors`, `Error Summary`.
- For each data row, map source columns to the first 8 columns. Use the header inspection from Step 0 to build the mapping. If source headers differ in casing/spacing, map by semantics.
- Compute the 4 derived columns as concrete values (int 0/1 and text strings):
  - `Price Error` = 1 if `Register Price != Promo Price`, else 0.
  - `Window Error` = 1 if `Sale Date < Promo Start Date` or `Sale Date > Promo End Date`, else 0.
    - For date comparison: if values are datetime objects, compare directly. If they are strings, parse with `datetime.strptime` using the format observed in Step 0. Be careful with None/missing values—treat missing dates conservatively (flag as error or skip based on what makes sense, but document the choice).
  - `Total Errors` = `Price Error + Window Error` (integer).
  - `Error Summary` = one of exactly: `None`, `Price Error`, `Window Error`, `Price Error, Window Error` — built from which flags are 1.
- Write all values as static Python values (int, str, datetime), NOT Excel formulas.

### 1c) `Summary` sheet
- Create sheet named exactly `Summary`.
- Row 1 headers: `SKU`, `Store ID`, `Price Errors`, `Window Errors`, `Total Errors`.
- Aggregate from the Formatted Data rows by `(SKU, Store ID)`: sum `Price Error`, sum `Window Error`, sum `Total Errors`.
- Filter: include only groups where summed `Total Errors > 0`.
- Sort ascending by `SKU` first, then `Store ID`.
- Append a final Grand Total row: `SKU`=`Grand Total`, `Store ID`=`-`, and the three numeric columns = dataset-wide totals (sum across ALL formatted data rows, not just the filtered groups — but since filtered groups already contain all error rows, summing the filtered groups gives the same result; verify this).
- Save the workbook.

### Validation checks before saving:
- Assert sheet names are exactly `['RawData', 'Formatted Data', 'Summary']`.
- Assert `Formatted Data` has 12 columns and same data-row count as `RawData`.
- Assert `Summary` last row has `Grand Total` in column A.
- Print the first 3 data rows of each sheet and the Grand Total row for visual confirmation.

## Step 2 – Build `/root/Promo_Register_Brief.docx`

Use `python-docx`.

- Compute from the Formatted Data: total_price_errors, total_window_errors, total_errors (sums of the respective columns across all data rows).
- Identify the top 2+ SKUs by total errors (aggregate Total Errors by SKU, sort descending, pick top 2 or more).
- Write a single paragraph (3-6 sentences) that includes:
  1. Definition: "A Price Error occurs when the register price does not match the promotional price. A Window Error occurs when a sale is recorded outside the promotional window (before the start date or after the end date)."
  2. Totals: "Across the dataset, there were {total_price_errors} Price Errors, {total_window_errors} Window Errors, and {total_errors} Total Errors."
  3. High-priority SKUs: "The SKUs with the most frequent exceptions were {sku1} and {sku2}, which should be prioritized for review."
  4. Recommendation: "We recommend re-validating promotional price uploads and tightening date-window controls in the POS system to prevent future discrepancies."
- Save as `/root/Promo_Register_Brief.docx`.

## Step 3 – Final verification
- Re-open both files and print key stats to confirm:
  - Excel sheet names
  - Row counts per sheet
  - Grand Total row values
  - Word document paragraph text (first 200 chars)

Execute all steps. If Step 0 reveals unexpected column names or date formats, adapt the mapping accordingly before proceeding.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=medium, tags=[excel, openpyxl, docx, audit, pricing].
Verifier config: timeout_sec=900.0.