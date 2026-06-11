# Task Instruction

Write and execute a Python script that performs all of the following steps:

## 1. Read the source workbook
- Open `/root/Promo_Price_Check_Source.xlsx` using `openpyxl` (to inspect sheet names and structure) and `pandas` (for data manipulation).
- Dynamically identify the correct sheet (there may be only one) and read all rows.
- Identify columns using case-insensitive matching for: `Promo ID`, `SKU`, `Promo Price`, `Register Price`, `Promo Start Date`, `Sale Date`, `Promo End Date`, `Store ID`. Map them to the canonical names.

## 2. Build the `RawData` sheet
- Copy the source data exactly (all columns, all rows, same order) into a DataFrame for the `RawData` sheet.

## 3. Build the `Formatted Data` sheet
- Start with the 8 canonical columns in the exact order listed above, preserving original row order.
- Normalize all date columns (`Promo Start Date`, `Sale Date`, `Promo End Date`) to `datetime` objects. Handle any string dates, Excel serial dates, or mixed formats robustly.
- Compute four new columns with **concrete values** (not formulas):
  - `Price Error`: 1 if `Register Price != Promo Price`, else 0 (compare as floats with tolerance or exact match as appropriate).
  - `Window Error`: 1 if `Sale Date < Promo Start Date` or `Sale Date > Promo End Date`, else 0. Compare as dates only (ignore time components).
  - `Total Errors`: `Price Error + Window Error`.
  - `Error Summary`: exactly one of `"None"`, `"Price Error"`, `"Window Error"`, `"Price Error, Window Error"` based on which flags are set.

## 4. Build the `Summary` sheet
- Group `Formatted Data` by `(SKU, Store ID)`.
- Sum `Price Error` → `Price Errors`, `Window Error` → `Window Errors`, `Total Errors` → `Total Errors` per group.
- Keep only groups where `Total Errors > 0`.
- Sort by `SKU` ascending, then `Store ID` ascending.
- Append a `Grand Total` row: `SKU` = `"Grand Total"`, `Store ID` = `"-"`, remaining columns = dataset-wide sums of Price Errors, Window Errors, Total Errors.
- The final column headers must be exactly: `SKU`, `Store ID`, `Price Errors`, `Window Errors`, `Total Errors`.

## 5. Write `/root/Promo_Register_Audit.xlsx`
- Use `openpyxl` (or pandas ExcelWriter with openpyxl engine) to write all three sheets: `RawData`, `Formatted Data`, `Summary` — with exactly those sheet names.
- Write concrete values everywhere (no Excel formulas). Ensure dates are written as date-formatted values or strings consistently.

## 6. Create `/root/Promo_Register_Brief.docx`
- Use `python-docx` to create a Word document.
- Write a short executive summary (3–6 sentences) that includes:
  - A plain-language definition of both checks: Price Error means the register price did not match the promotional price; Window Error means the sale occurred outside the promotional window.
  - The computed totals for Price Errors, Window Errors, and Total Errors (use the actual numbers from the data).
  - At least one actionable recommendation (e.g., "Implement automated register-price synchronization...").
  - Mention at least two specific high-priority SKUs that had the most frequent exceptions (determine these from the Summary data).

## 7. Validation
- After writing both files, re-open `/root/Promo_Register_Audit.xlsx` and verify:
  - Sheet names are exactly `['RawData', 'Formatted Data', 'Summary']`.
  - `Formatted Data` has 12 columns with the correct headers.
  - `Summary` last row has SKU == `"Grand Total"`.
  - Print the total Price Errors, Window Errors, Total Errors from the Grand Total row.
- Confirm `/root/Promo_Register_Brief.docx` exists and print its text content.

Install any needed packages (`pip install openpyxl pandas python-docx`) before running the script. Execute the entire script in one go and show the output.

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