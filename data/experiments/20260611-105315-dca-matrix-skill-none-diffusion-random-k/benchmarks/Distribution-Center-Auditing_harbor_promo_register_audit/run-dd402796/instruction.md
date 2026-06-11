# Task Instruction

## Task: Build Promo Register Audit Files

You need to create two output files from a source workbook. Follow these steps precisely.

### Step 1: Inspect the source file
```bash
cd /root
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('Promo_Price_Check_Source.xlsx')
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'Sheet: {sn}, rows={ws.max_row}, cols={ws.max_column}')
    for row in ws.iter_rows(min_row=1, max_row=min(5, ws.max_row), values_only=False):
        print([cell.value for cell in row])
"
```
Note the exact column names and their order. Also check data types (are dates datetime objects or strings?).

### Step 2: Create both output files with a single Python script

Write and run a Python script `/root/build_audit.py` that does the following:

#### A) Read source data
- Use `openpyxl` to read the source workbook. Identify the sheet (likely the first/only sheet).
- Load all rows into a list of dicts or a pandas DataFrame.
- Map columns to the 8 required columns: `Promo ID`, `SKU`, `Promo Price`, `Register Price`, `Promo Start Date`, `Sale Date`, `Promo End Date`, `Store ID`. If the source column names differ slightly, map them correctly.

#### B) Build `Promo_Register_Audit.xlsx` with exactly 3 sheets:

**Sheet 1: `RawData`**
- Copy the source table exactly as-is (same columns, same order, same values).

**Sheet 2: `Formatted Data`**
- Same rows in same order as RawData.
- First 8 columns exactly as listed above (use those exact header strings).
- Add 4 computed columns (columns 9-12) with these exact headers:
  - `Price Error`: 1 if `Register Price != Promo Price`, else 0 (compare as numbers)
  - `Window Error`: 1 if `Sale Date < Promo Start Date` or `Sale Date > Promo End Date`, else 0 (compare as dates; ensure all date values are converted to `datetime.date` or `datetime.datetime` for comparison)
  - `Total Errors`: `Price Error + Window Error`
  - `Error Summary`: exactly one of these strings:
    - `"None"` (if both errors are 0)
    - `"Price Error"` (if only price error)
    - `"Window Error"` (if only window error)
    - `"Price Error, Window Error"` (if both errors)
- Write concrete integer values (0 or 1) for the error columns, not Excel formulas.
- Write concrete strings for Error Summary.

**Sheet 3: `Summary`**
- Headers exactly: `SKU`, `Store ID`, `Price Errors`, `Window Errors`, `Total Errors`
- Group the Formatted Data by (SKU, Store ID).
- Sum `Price Error` → `Price Errors`, `Window Error` → `Window Errors`, `Total Errors` → `Total Errors` for each group.
- Include ONLY groups where `Total Errors > 0`.
- Sort by `SKU` ascending, then `Store ID` ascending.
- Append a final row: `SKU`=`"Grand Total"`, `Store ID`=`"-"`, and the remaining 3 columns = overall dataset totals (sum across ALL rows in Formatted Data, not just filtered groups — though they should be the same since groups with 0 errors contribute 0).

#### C) Build `Promo_Register_Brief.docx`
- Use `python-docx` to create the document.
- Write a short executive summary paragraph (3-6 sentences) that includes:
  1. Plain-language definition of Price Error (register price doesn't match promo price) and Window Error (sale occurred outside the promotional window).
  2. The computed grand totals for Price Errors, Window Errors, and Total Errors (use actual numbers from the data).
  3. At least one actionable recommendation.
  4. Mention at least two specific SKUs that have the highest error counts (look at the Summary sheet data to find these).

### Step 3: Validate outputs
After running the script, verify:
```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('Promo_Register_Audit.xlsx')
print('Sheets:', wb.sheetnames)
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'\n--- {sn} --- rows={ws.max_row}, cols={ws.max_column}')
    for row in ws.iter_rows(min_row=1, max_row=min(4, ws.max_row), values_only=True):
        print(row)
    if ws.max_row > 4:
        print('...')
        for row in ws.iter_rows(min_row=ws.max_row-1, max_row=ws.max_row, values_only=True):
            print(row)
"
```
Also verify the docx:
```bash
python3 -c "
from docx import Document
doc = Document('Promo_Register_Brief.docx')
for p in doc.paragraphs:
    print(p.text)
"
```

Check:
- Sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
- Formatted Data has exactly 12 columns with correct headers.
- Summary has exactly 5 columns with correct headers.
- Summary last row has SKU="Grand Total" and Store ID="-".
- Error Summary values are exactly one of the 4 allowed strings.
- Price Error and Window Error columns contain integers 0 or 1.
- The docx mentions both error types, totals, a recommendation, and at least 2 SKUs.

### Important Notes
- Install any needed packages: `pip install openpyxl python-docx pandas` if not already available.
- Date comparison: ensure you handle date types consistently. If dates come as datetime objects from openpyxl, compare them directly. If they come as strings, parse them first.
- Store ID might be numeric; preserve its original type in RawData but ensure it works for sorting in Summary.
- Output filenames must be exactly `/root/Promo_Register_Audit.xlsx` and `/root/Promo_Register_Brief.docx`.

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