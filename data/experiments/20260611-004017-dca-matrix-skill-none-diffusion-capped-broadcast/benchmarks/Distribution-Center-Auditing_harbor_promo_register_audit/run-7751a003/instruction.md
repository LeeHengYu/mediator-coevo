# Task Instruction

## Task: Build Promo Register Audit Workbook and Brief

You must create two files:
1. `/root/Promo_Register_Audit.xlsx`
2. `/root/Promo_Register_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect the source
- Open and read `/root/Promo_Price_Check_Source.xlsx` using `openpyxl` (or pandas). Print the sheet names, the header row, the first 5 data rows, the total row count, and the dtypes/sample values of every column. This is critical before writing any logic.
- Identify the exact column names as they appear in the source (they should map to: Promo ID, SKU, Promo Price, Register Price, Promo Start Date, Sale Date, Promo End Date, Store ID — but verify exact spelling/casing).
- Check whether date columns are actual datetime objects or strings. Print a few sample values with their Python types.

#### Step 1: Create `RawData` sheet
- Copy the source data exactly (all rows, all columns, same order) into a sheet named `RawData` in `/root/Promo_Register_Audit.xlsx`.

#### Step 2: Create `Formatted Data` sheet
- Start with the same data and row order as RawData.
- The first 8 columns must have exactly these headers (rename if needed):
  1. `Promo ID`
  2. `SKU`
  3. `Promo Price`
  4. `Register Price`
  5. `Promo Start Date`
  6. `Sale Date`
  7. `Promo End Date`
  8. `Store ID`
- Add 4 new computed columns (9–12) with exactly these headers:
  9. `Price Error`
  10. `Window Error`
  11. `Total Errors`
  12. `Error Summary`

**Computation rules (write concrete values, NOT Excel formulas):**
- `Price Error` = 1 if `Register Price` != `Promo Price`, else 0. Compare numerically (convert to float if needed; handle floating point by rounding to 2 decimal places before comparing).
- `Window Error` = 1 if `Sale Date` < `Promo Start Date` OR `Sale Date` > `Promo End Date`, else 0. Convert all date columns to `datetime.date` or `pandas.Timestamp` for comparison. If any date is a string, parse it. Be careful with time components — compare dates only (strip time if present).
- `Total Errors` = `Price Error` + `Window Error` (integer).
- `Error Summary` = exactly one of these strings:
  - `"None"` (if both errors are 0)
  - `"Price Error"` (if only price error)
  - `"Window Error"` (if only window error)
  - `"Price Error, Window Error"` (if both errors)

**Important:** Write these as concrete int/str values in the cells. Do NOT use Excel formulas.

#### Step 3: Create `Summary` sheet
- Headers must be exactly: `SKU`, `Store ID`, `Price Errors`, `Window Errors`, `Total Errors`
- Group the `Formatted Data` by `(SKU, Store ID)`.
- Sum `Price Error`, `Window Error`, and `Total Errors` for each group.
- Include ONLY groups where the summed `Total Errors > 0`.
- Sort by `SKU` ascending, then `Store ID` ascending. (If SKU values are numeric, sort numerically; if strings, sort lexicographically. Print the SKU dtype to decide.)
- Append a final row: `SKU` = `"Grand Total"`, `Store ID` = `"-"`, and the remaining 3 columns = the sum across all included rows (i.e., dataset-wide totals of Price Errors, Window Errors, Total Errors).

#### Step 4: Save the Excel file
- Ensure the workbook `/root/Promo_Register_Audit.xlsx` has exactly three sheets in this order: `RawData`, `Formatted Data`, `Summary`.
- Save and close.

#### Step 5: Verify the Excel output
- Re-open `/root/Promo_Register_Audit.xlsx` and for each sheet:
  - Print sheet name, headers, row count, and first 3 + last 3 data rows.
- For `Formatted Data`: print value counts of `Error Summary` column, and sum of `Price Error`, `Window Error`, `Total Errors`.
- For `Summary`: print all rows including the Grand Total row. Verify the Grand Total row values match the sums from Formatted Data.

#### Step 6: Create `/root/Promo_Register_Brief.docx`
- Use `python-docx` to create a Word document.
- Write a short executive summary (3–6 sentences) that includes ALL of the following:
  1. A plain-language definition of both checks:
     - Price Error: when the register price does not match the promotional price.
     - Window Error: when a sale occurs outside the promotional window (before start or after end date).
  2. The exact computed totals for Price Errors, Window Errors, and Total Errors (use the Grand Total values from the Summary sheet).
  3. At least one actionable recommendation (e.g., "Implement automated price verification at POS" or "Review promotional calendar sync processes").
  4. Mention at least two specific high-priority SKUs that have the most frequent exceptions (look at the Summary sheet to identify the top 2 SKUs by Total Errors).
- Save the file.

#### Step 7: Final verification
- Confirm both files exist: `/root/Promo_Register_Audit.xlsx` and `/root/Promo_Register_Brief.docx`.
- Re-read the docx and print its text to verify all required elements are present.
- Print "TASK COMPLETE" when done.

### Technical Notes
- Install any needed packages: `pip install openpyxl python-docx pandas` if not already available.
- Use pandas for data manipulation but openpyxl for writing the Excel file (to ensure concrete values, not formulas, and to control sheet names precisely).
- When writing with openpyxl or pandas ExcelWriter, make sure date columns are written as dates (not strings) in RawData and Formatted Data, matching the source format.
- Double-check that Store ID values are preserved in their original type (don't accidentally convert int store IDs to float).
- The `Summary` sheet column headers are `Price Errors`, `Window Errors`, `Total Errors` (plural) — different from the `Formatted Data` headers `Price Error`, `Window Error` (singular). Be precise.

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