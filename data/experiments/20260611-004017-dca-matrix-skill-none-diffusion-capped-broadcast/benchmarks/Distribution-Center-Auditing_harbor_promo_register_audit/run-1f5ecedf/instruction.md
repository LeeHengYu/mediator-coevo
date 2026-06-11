# Task Instruction

Create the two deliverable files `/root/Promo_Register_Audit.xlsx` and `/root/Promo_Register_Brief.docx` by executing a Python script. Follow these steps precisely:

## Step 1: Inspect the source file
- Read `/root/Promo_Price_Check_Source.xlsx` to understand its structure: sheet names, column headers, data types, number of rows. Print the first few rows and dtypes.

## Step 2: Build the Python script
Write and execute a single Python script that does everything below.

### Reading the source
- Use `openpyxl` to read the source workbook directly (not pandas) to preserve exact cell values and types. Read all rows from the first (or only) sheet into a list of lists, with the first row as headers.
- Alternatively, if using pandas: read with `pd.read_excel(..., dtype=str)` to keep everything as strings initially, preventing any automatic type conversion. Then fill any NaN values with the original cell content (or empty string if truly empty). **Critical**: Do NOT let pandas parse dates or convert types.

### RawData sheet
- Write the source data exactly as-is into the `RawData` sheet of `/root/Promo_Register_Audit.xlsx`. Every cell must match the source cell value exactly — strings stay strings, numbers stay numbers. If using pandas with `dtype=str`, convert numeric-looking columns back to numeric where appropriate, but keep date columns as strings.
- **Key requirement from feedback**: Date columns (`Promo Start Date`, `Sale Date`, `Promo End Date`) MUST be written as strings (e.g., '2026-03-01'), NOT as datetime objects. Also ensure any NaN/None values are preserved as whatever the source file contains (e.g., 'N/A' stays 'N/A', empty stays empty).

### Formatted Data sheet
- Copy all rows in the same order as RawData.
- Keep the first 8 columns exactly: Promo ID, SKU, Promo Price, Register Price, Promo Start Date, Sale Date, Promo End Date, Store ID.
- Add four new columns (9-12):
  - **Price Error**: Compare `Register Price` and `Promo Price` as numbers. If they differ, 1; else 0.
  - **Window Error**: Parse `Sale Date`, `Promo Start Date`, `Promo End Date` as dates for comparison only. If `Sale Date < Promo Start Date` or `Sale Date > Promo End Date`, then 1; else 0.
  - **Total Errors**: `Price Error + Window Error` (integer).
  - **Error Summary**: Exactly one of: `None`, `Price Error`, `Window Error`, `Price Error, Window Error` (use the string `None` not Python None).
- Write these computed columns as concrete values (integers for error flags, strings for Error Summary).
- **Date columns in Formatted Data must also be written as strings**, not datetime objects.

### Summary sheet
- Headers exactly: SKU, Store ID, Price Errors, Window Errors, Total Errors
- Group Formatted Data by (SKU, Store ID). Sum Price Error, Window Error, Total Errors for each group.
- Include only groups where Total Errors > 0.
- Sort by SKU ascending, then Store ID ascending.
- Append a final row: SKU='Grand Total', Store ID='-', and the remaining columns are the dataset-wide totals (sum across ALL rows of Formatted Data, not just the filtered groups).

### Writing the Excel file
- Use `openpyxl` for writing. Create the workbook with sheets named exactly `RawData`, `Formatted Data`, `Summary`.
- When writing date columns, ensure they are written as plain strings. If using openpyxl directly, just write the string value. If using pandas ExcelWriter, convert date columns to string type before writing.
- Write numeric values (Promo Price, Register Price, error flags, totals) as actual numbers (int/float), not strings.
- Do NOT use pandas `to_excel` with default settings that might convert types. If you do use pandas, set the engine to openpyxl and ensure date columns are string dtype.

### Word document
- Use `python-docx` to create `/root/Promo_Register_Brief.docx`.
- Write 3-6 sentences covering:
  1. Definition of Price Error (register price doesn't match promo price) and Window Error (sale occurred outside the promotional window).
  2. The computed totals for Price Errors, Window Errors, and Total Errors (use the Grand Total values).
  3. At least one actionable recommendation.
  4. Mention at least two specific SKUs that have the highest error counts.

## Step 3: Validate
After running the script:
1. Re-read `/root/Promo_Register_Audit.xlsx` and print sheet names.
2. Print the first 3 rows of each sheet to verify content.
3. Print the type of a date cell value from RawData to confirm it's a string, not datetime.
4. Print the last row of Summary to confirm Grand Total row.
5. Confirm `/root/Promo_Register_Brief.docx` exists and print its text content.

## Critical Reminders
- Date columns MUST be strings in the output Excel, not datetime objects.
- NaN/None values from source must be preserved exactly as they appear in the source (read with openpyxl directly to see actual cell values).
- Sheet names must be exactly `RawData`, `Formatted Data`, `Summary`.
- Error Summary string `None` means the text string 'None', not a null value.
- The Grand Total row's error sums should be computed from ALL rows in Formatted Data, not just the filtered error rows.

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