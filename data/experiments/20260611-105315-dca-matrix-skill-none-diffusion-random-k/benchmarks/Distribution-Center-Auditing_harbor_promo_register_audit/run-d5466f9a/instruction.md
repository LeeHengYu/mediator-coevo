# Task Instruction

## Task: Build Promo Register Audit Deliverables

You must create two files from the source workbook `/root/Promo_Price_Check_Source.xlsx`:
1. `/root/Promo_Register_Audit.xlsx`
2. `/root/Promo_Register_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect the source
- Read `/root/Promo_Price_Check_Source.xlsx` using Python (openpyxl or pandas).
- Print the sheet names, column headers, first 5 rows, dtypes, and row count.
- Identify the exact column names for: Promo ID, SKU, Promo Price, Register Price, Promo Start Date, Sale Date, Promo End Date, Store ID.
- Check if dates are actual datetime objects or strings; convert to dates if needed.

#### Step 1: Build `RawData` sheet
- Copy the entire source table exactly (all rows, all columns, same order) into a sheet named `RawData` in `/root/Promo_Register_Audit.xlsx`.

#### Step 2: Build `Formatted Data` sheet
- Start with the same data and row order as RawData.
- Keep the first 8 columns exactly as: `Promo ID`, `SKU`, `Promo Price`, `Register Price`, `Promo Start Date`, `Sale Date`, `Promo End Date`, `Store ID`.
- Add four new computed columns (9-12) with EXACTLY these headers: `Price Error`, `Window Error`, `Total Errors`, `Error Summary`.
- Compute concrete values (not Excel formulas) for each row:
  - `Price Error` = 1 if `Register Price` != `Promo Price`, else 0. Use numeric comparison (handle floating point: consider values equal if they match to 2 decimal places or use exact comparison as appropriate given the data).
  - `Window Error` = 1 if `Sale Date` < `Promo Start Date` OR `Sale Date` > `Promo End Date`, else 0. Compare as date objects (strip any time component, compare calendar dates only).
  - `Total Errors` = `Price Error` + `Window Error` (integer).
  - `Error Summary` = exactly one of these four strings:
    - `"None"` if both errors are 0
    - `"Price Error"` if only Price Error is 1
    - `"Window Error"` if only Window Error is 1
    - `"Price Error, Window Error"` if both are 1
- Write all values as concrete numbers/strings (not formulas).

#### Step 3: Build `Summary` sheet
- Aggregate from the Formatted Data by (SKU, Store ID).
- For each group, sum `Price Errors` (= sum of Price Error), `Window Errors` (= sum of Window Error), `Total Errors` (= sum of Total Errors).
- Include ONLY groups where `Total Errors > 0`.
- Sort by `SKU` ascending, then `Store ID` ascending.
- Headers must be exactly: `SKU`, `Store ID`, `Price Errors`, `Window Errors`, `Total Errors`.
- Append a final row: SKU = `Grand Total`, Store ID = `-`, and the remaining three columns = the dataset-wide totals (sum across ALL rows in Formatted Data, not just the filtered ones — actually, since the grand total should equal the sum of the displayed rows because we only excluded groups with 0 total errors, verify this).
- IMPORTANT: The Grand Total row sums should be the totals across the ENTIRE dataset (all rows from Formatted Data), not just the filtered summary rows. Verify: sum of Price Error across all Formatted Data rows, sum of Window Error across all Formatted Data rows, sum of Total Errors across all Formatted Data rows.

#### Step 4: Ensure sheet order
- The workbook must have exactly three sheets in this order: `RawData`, `Formatted Data`, `Summary`.

#### Step 5: Build Word document `/root/Promo_Register_Brief.docx`
- Use python-docx to create the file.
- Write a short executive summary (3-6 sentences) that includes:
  - A plain-language definition of both checks: Price Error (register price doesn't match the promotional price) and Window Error (sale occurred outside the promotional window dates).
  - The exact computed totals for Price Errors, Window Errors, and Total Errors (use the numbers from your Grand Total row).
  - At least one actionable recommendation (e.g., implement automated price verification, retrain staff, audit POS systems).
  - Mention at least two specific high-priority SKUs that have the most frequent exceptions (look at your Summary data to identify the top 2 SKUs by total errors).

#### Step 6: Validation
- Re-open `/root/Promo_Register_Audit.xlsx` and verify:
  - Sheet names are exactly `['RawData', 'Formatted Data', 'Summary']`.
  - `RawData` row count matches source.
  - `Formatted Data` has 12 columns with correct headers.
  - `Summary` last row has SKU = `Grand Total` and Store ID = `-`.
  - Print a few sample rows from Formatted Data showing error computations.
  - Print the full Summary sheet.
- Re-open `/root/Promo_Register_Brief.docx` and print its text to verify content.

Use `openpyxl` for Excel writing (to support multiple named sheets) and `python-docx` for the Word document. Install any missing packages with pip if needed.

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