# Task Instruction

Create two output files: `/root/Promo_Register_Audit.xlsx` and `/root/Promo_Register_Brief.docx`.

## Step-by-step plan

### Step 0 — Inspect the source
```python
import openpyxl
wb = openpyxl.load_workbook('/root/Promo_Price_Check_Source.xlsx')
for name in wb.sheetnames:
    ws = wb[name]
    print(f'Sheet: {name}, rows={ws.max_row}, cols={ws.max_column}')
    for r in ws.iter_rows(min_row=1, max_row=min(5, ws.max_row), values_only=True):
        print(r)
```
Examine the exact column headers and the **Python types** of every cell (especially dates — are they strings or datetime objects?). Print `type(cell)` for each cell in row 2 so you know exactly what the source contains.

### Step 1 — Read source data into Python lists
Using openpyxl (do NOT use pandas for reading — keep raw cell values):
- Read the first (or only) sheet of `Promo_Price_Check_Source.xlsx`.
- Row 1 → headers list.
- Rows 2+ → list of lists, each list preserving the exact Python objects openpyxl returns (strings, ints, floats, datetimes — whatever they are).

Define `BASE_HEADERS` = the first 8 header strings exactly as they appear in the source.

### Step 2 — Build `Promo_Register_Audit.xlsx`

Create a new workbook with openpyxl. Remove the default sheet after creating the three required sheets.

#### Sheet `RawData`
- Write the header row, then every data row, using the **exact same values and types** as read from the source. Do not convert, format, or coerce anything.

#### Sheet `Formatted Data`
- Write the same 8-column header row, then append 4 new headers: `Price Error`, `Window Error`, `Total Errors`, `Error Summary`.
- For each data row, write the first 8 values **exactly as they appear in the source** (same type, same value — do NOT convert strings to datetimes or datetimes to strings). This is critical: the verifier asserts `out[:8] == [source_row[h] for h in BASE_HEADERS]`.
- Compute the 4 new columns:
  - Map headers to indices: find indices for `Promo Price`, `Register Price`, `Promo Start Date`, `Sale Date`, `Promo End Date`.
  - `Price Error`: Compare `Register Price` and `Promo Price`. Convert both to float before comparing (handles cases where one is int and the other float). If they are not equal → 1, else → 0.
  - `Window Error`: Compare dates. If the values are strings, parse them with `datetime.strptime(val, '%Y-%m-%d')` (or whichever format you observe) for comparison only — but still write the original value in columns 1-8. If they are already datetime objects, compare directly. `Window Error` = 1 if `Sale Date < Promo Start Date` or `Sale Date > Promo End Date`, else 0.
  - `Total Errors` = `Price Error + Window Error` (write as int).
  - `Error Summary`: string, exactly one of `None`, `Price Error`, `Window Error`, `Price Error, Window Error`.
- Write all 4 computed columns as concrete values (int for the numeric ones, string for Error Summary). Do NOT write Excel formulas.

#### Sheet `Summary`
- Headers: `SKU`, `Store ID`, `Price Errors`, `Window Errors`, `Total Errors`.
- Aggregate from the Formatted Data rows by `(SKU, Store ID)`: sum `Price Error` and `Window Error` per group; `Total Errors` = sum of both.
- Keep only groups where `Total Errors > 0`.
- Sort by `SKU` ascending then `Store ID` ascending. For correct sorting: if SKU/Store ID are numeric, sort numerically; if strings, sort lexicographically.
- Append a final `Grand Total` row: `SKU`=`Grand Total`, `Store ID`=`-`, then the column sums across the **entire dataset** (not just filtered groups — sum Price Errors, Window Errors, Total Errors from ALL Formatted Data rows).

Save to `/root/Promo_Register_Audit.xlsx`.

### Step 3 — Build `/root/Promo_Register_Brief.docx`
Using `python-docx`:
- Add a heading "Promo Register Audit – Executive Summary".
- Write 3-6 sentences that include:
  1. Plain-language definition of Price Error (register price differs from promotional price) and Window Error (sale occurred outside the promotional window).
  2. The exact computed totals: "The audit identified X Price Errors, Y Window Errors, and Z Total Errors."
  3. Mention at least two specific SKUs with the highest error counts (pick the top 2 SKUs by total errors from the Summary sheet).
  4. At least one actionable recommendation (e.g., "Recommend re-syncing POS registers with the promo pricing database before each promotional period").
- Save to `/root/Promo_Register_Brief.docx`.

### Step 4 — Validate
- Reopen `/root/Promo_Register_Audit.xlsx` with openpyxl.
- Confirm sheet names are exactly `['RawData', 'Formatted Data', 'Summary']`.
- Confirm `Formatted Data` row 2 columns 1-8 match the source row 2 columns 1-8 in both value and type.
- Confirm `Summary` last row starts with `Grand Total`.
- Confirm `/root/Promo_Register_Brief.docx` exists and contains text.
- Print confirmation of all checks.

## Critical reminders
- **Do NOT convert date values when writing the first 8 columns of Formatted Data.** Write the exact same Python objects that openpyxl returned from the source. The verifier compares them cell-by-cell.
- Write computed columns as literal values, not formulas.
- Sheet names must be exact (case-sensitive, spacing matters).
- File paths must be exact.
- Use `openpyxl` for Excel and `python-docx` for Word. Install with pip if needed.

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