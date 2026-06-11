# Task Instruction

Execute the following task by writing and running a Python script.

## Goal
Read `/root/Promo_Price_Check_Source.xlsx` and produce two deliverables:
1. `/root/Promo_Register_Audit.xlsx` (3 sheets: `RawData`, `Formatted Data`, `Summary`)
2. `/root/Promo_Register_Brief.docx` (executive summary)

## Steps

### Step 1: Inspect the source file
Read `/root/Promo_Price_Check_Source.xlsx` using openpyxl or pandas to understand the columns, data types, and row count. Print the first few rows and all column headers.

### Step 2: Write a Python script that does the following

Use `openpyxl` for Excel and `python-docx` for Word. Install them if needed (`pip install openpyxl python-docx`).

#### Sheet 1: `RawData`
- Copy the entire source table exactly as-is (preserve all values and types, including dates as datetime objects).

#### Sheet 2: `Formatted Data`
- Same row order as RawData.
- First 8 columns copied exactly preserving original Python types (especially keep dates as datetime objects, numbers as numbers):
  1. Promo ID
  2. SKU
  3. Promo Price
  4. Register Price
  5. Promo Start Date
  6. Sale Date
  7. Promo End Date
  8. Store ID
- Add 4 new columns (9-12) with these exact headers and concrete computed values (NOT formulas):
  - **Price Error**: 1 if float(Register Price) != float(Promo Price), else 0
  - **Window Error**: 1 if Sale Date < Promo Start Date OR Sale Date > Promo End Date, else 0. Convert to datetime for comparison if needed, but compare as dates (calendar dates only, ignore time).
  - **Total Errors**: Price Error + Window Error (integer)
  - **Error Summary**: Exactly one of: `None`, `Price Error`, `Window Error`, `Price Error, Window Error`

IMPORTANT: For date comparisons, convert values to date objects (not datetime with time component) to ensure correct calendar-date comparison. For price comparisons, convert both to float to avoid type mismatch issues.

#### Sheet 3: `Summary`
- Headers exactly: `SKU`, `Store ID`, `Price Errors`, `Window Errors`, `Total Errors`
- Aggregate from Formatted Data by (SKU, Store ID) pair.
- Include ONLY groups where Total Errors > 0.
- Sort by SKU ascending, then Store ID ascending.
- Append a final row: SKU=`Grand Total`, Store ID=`-`, and the remaining columns are the dataset-wide totals of Price Errors, Window Errors, Total Errors.

#### Word Document: `/root/Promo_Register_Brief.docx`
- 3-6 sentence executive summary paragraph.
- Include plain-language definitions of both checks:
  - Price Error: when the register price doesn't match the promotional price.
  - Window Error: when a sale occurs outside the promotional window (before start or after end date).
- Include the computed totals for Price Errors, Window Errors, and Total Errors (use exact numbers from the data).
- Include at least one actionable recommendation.
- Mention at least two high-priority SKUs with the most frequent exceptions (identify from the data by summing Total Errors per SKU and picking the top 2).

### Step 3: Run the script and verify
- After running, verify the output by reading back both files:
  - Print sheet names of the Excel file.
  - Print row counts for each sheet.
  - Print the first 3 and last 3 rows of `Formatted Data`.
  - Print all rows of `Summary`.
  - Print the full text of the Word document.
- Confirm worksheet names are exactly `RawData`, `Formatted Data`, `Summary`.
- Confirm the Grand Total row exists and values are correct.

### Key Implementation Notes (from prior successful execution)
- When copying the first 8 columns to Formatted Data, preserve the exact Python types from the source (datetime stays datetime, int stays int, etc.). Do NOT convert dates to strings.
- For the error logic columns, perform type-safe comparisons: convert to float for price comparison, convert to date for date comparison.
- Write integer values (0 or 1) for Price Error and Window Error, not booleans.
- Write the string `None` (not Python None) for Error Summary when there are no errors.
- For the Summary sheet, Store ID should be preserved in its original type (likely integer). The Grand Total row's Store ID should be the string `-`.

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