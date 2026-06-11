# Task Instruction

## Task: Promo Register Audit

You must create two deliverable files from a source workbook. Follow every step carefully.

### Step 1: Inspect the Source

Read `/root/Promo_Price_Check_Source.xlsx` to understand its structure:
```python
import openpyxl
wb = openpyxl.load_workbook('/root/Promo_Price_Check_Source.xlsx')
for name in wb.sheetnames:
    ws = wb[name]
    print(f'Sheet: {name}, rows={ws.max_row}, cols={ws.max_column}')
    for row in ws.iter_rows(min_row=1, max_row=min(5, ws.max_row), values_only=True):
        print(row)
```
Identify the column positions for: Promo ID, SKU, Promo Price, Register Price, Promo Start Date, Sale Date, Promo End Date, Store ID. Note the exact header names used in the source.

### Step 2: Build `/root/Promo_Register_Audit.xlsx`

Use `openpyxl` (or pandas + openpyxl engine) to create the workbook with exactly three sheets named: `RawData`, `Formatted Data`, `Summary`.

#### Sheet: `RawData`
- Copy every row (headers + data) from the source workbook exactly as-is. Preserve all values including dates. Do not alter column order, types, or content.

#### Sheet: `Formatted Data`
- Same rows and row order as RawData.
- First 8 columns must have exactly these headers (rename if the source headers differ):
  1. `Promo ID`
  2. `SKU`
  3. `Promo Price`
  4. `Promo End Date`... 

**IMPORTANT: The required column order is precisely:**
  1. `Promo ID`
  2. `SKU`
  3. `Promo Price`
  4. `Register Price`
  5. `Promo Start Date`
  6. `Sale Date`
  7. `Promo End Date`
  8. `Store ID`

If the source columns are in a different order, rearrange them to match this exact order. Map the source columns by their header names (case-insensitive matching if needed).

- Add 4 new columns (9-12) with exactly these headers:
  9. `Price Error`
  10. `Window Error`
  11. `Total Errors`
  12. `Error Summary`

- Compute values as concrete numbers/strings (NOT Excel formulas):
  - `Price Error` = 1 if `Register Price` != `Promo Price`, else 0
  - `Window Error` = 1 if `Sale Date` < `Promo Start Date` OR `Sale Date` > `Promo End Date`, else 0
    - When comparing dates, convert all date values to `datetime.date` or `datetime.datetime` objects for reliable comparison. If any date is stored as a string, parse it. If stored as a datetime, use `.date()` for comparison.
  - `Total Errors` = `Price Error` + `Window Error`
  - `Error Summary` = one of exactly:
    - `None` (the string, not Python None)
    - `Price Error`
    - `Window Error`
    - `Price Error, Window Error`
    Based on which errors are flagged (1).

#### Sheet: `Summary`
- Headers exactly: `SKU`, `Store ID`, `Price Errors`, `Window Errors`, `Total Errors`
- Group rows from `Formatted Data` by (SKU, Store ID).
- For each group, sum Price Error, Window Error, Total Errors.
- Include ONLY groups where the summed Total Errors > 0.
- Sort by SKU ascending (alphabetical/natural), then Store ID ascending.
- Append a final row: SKU=`Grand Total`, Store ID=`-`, and the remaining columns = grand totals across the entire dataset (sum of ALL rows from Formatted Data, not just the filtered ones... actually sum from the filtered summary rows since they already represent all error rows). 

**Clarification on Grand Total**: The Grand Total row should contain the sum of Price Errors, Window Errors, and Total Errors columns from all the summary rows above it (which equals the sum from the full Formatted Data since non-error groups contribute 0).

### Step 3: Build `/root/Promo_Register_Brief.docx`

Use `python-docx` to create the Word document.

Content requirements (3-6 sentences, as one or two paragraphs):
1. Define both checks in plain language:
   - Price Error: when the register price does not match the intended promotional price.
   - Window Error: when a sale occurs outside the valid promotional window (before start or after end date).
2. State the computed totals: total Price Errors, total Window Errors, and total combined Total Errors from the dataset.
3. Identify at least two specific SKUs that have the highest number of total errors (look at the Summary sheet data to find them).
4. Provide at least one actionable recommendation (e.g., retraining cashiers, auditing POS system sync, reviewing promotional calendar adherence).

### Step 4: Validate

After creating both files, verify:
1. Re-open `/root/Promo_Register_Audit.xlsx` and confirm:
   - Exactly 3 sheets with exact names: `RawData`, `Formatted Data`, `Summary`
   - `RawData` row count matches source
   - `Formatted Data` has 12 columns with correct headers
   - `Formatted Data` has the same number of data rows as RawData
   - Spot-check a few Price Error and Window Error calculations
   - `Summary` headers are exactly: `SKU`, `Store ID`, `Price Errors`, `Window Errors`, `Total Errors`
   - Summary last row has SKU=`Grand Total`
   - Grand Total's Total Errors = sum of all Total Errors in Formatted Data
2. Re-open `/root/Promo_Register_Brief.docx` and print its text to confirm it contains the required elements.

Print all validation results. If anything is wrong, fix it before finishing.

### Technical Notes
- Install any needed packages: `pip install openpyxl python-docx` if not already available.
- When writing dates to Excel with openpyxl, write them as datetime objects so they render properly.
- Be careful with date comparisons: ensure you're comparing like types (date to date, not date to datetime).

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