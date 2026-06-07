# Task Instruction

Build two deliverables from `/root/Promo_Price_Check_Source.xlsx`: an Excel audit workbook and a Word brief.

## Step 1: Inspect the source
Open `/root/Promo_Price_Check_Source.xlsx` with pandas/openpyxl. Identify the sheet name and confirm the columns include: Promo ID, SKU, Promo Price, Register Price, Promo Start Date, Sale Date, Promo End Date, Store ID. Note the row count and dtypes (especially dates).

## Step 2: Create `/root/Promo_Register_Audit.xlsx` with three sheets in this order: `RawData`, `Formatted Data`, `Summary`.

### Sheet `RawData`
- Copy the source table verbatim (same columns, same row order, same values). Preserve date values as dates.

### Sheet `Formatted Data`
- Same row order as RawData.
- Columns 1-8 exactly: `Promo ID`, `SKU`, `Promo Price`, `Register Price`, `Promo Start Date`, `Sale Date`, `Promo End Date`, `Store ID`.
- Add columns 9-12 with headers exactly: `Price Error`, `Window Error`, `Total Errors`, `Error Summary`.
- Compute concrete values (not formulas):
  - `Price Error` = 1 if `Register Price` != `Promo Price` else 0. Be careful with float comparison; round both to 2 decimals or use a tolerance like abs(diff) > 1e-6.
  - `Window Error` = 1 if `Sale Date` < `Promo Start Date` or `Sale Date` > `Promo End Date` else 0. Compare as dates (normalize to date, not datetime with time component).
  - `Total Errors` = `Price Error` + `Window Error` (integer 0, 1, or 2).
  - `Error Summary` exactly one of: `None`, `Price Error`, `Window Error`, `Price Error, Window Error` (note the comma-space separator and exact casing).

### Sheet `Summary`
- Headers exactly: `SKU`, `Store ID`, `Price Errors`, `Window Errors`, `Total Errors`.
- Group `Formatted Data` by `(SKU, Store ID)`, summing `Price Error`, `Window Error`, and `Total Errors`.
- Keep only groups where `Total Errors > 0`.
- Sort by `SKU` ascending, then `Store ID` ascending.
- Append a final row: `SKU` = `Grand Total`, `Store ID` = `-`, then dataset-wide sums of Price Errors, Window Errors, Total Errors (sums over the included groups, which equal totals over all rows since groups with 0 contribute 0).

## Step 3: Create `/root/Promo_Register_Brief.docx`
Use python-docx. Write a 3-6 sentence executive summary that includes:
- Plain-language definition of `Price Error` (register price did not match the promo price) and `Window Error` (the sale occurred outside the promo start/end date window).
- Computed totals for Price Errors, Window Errors, and Total Errors (use the dataset totals from the Summary grand total row).
- At least one actionable recommendation (e.g., audit register price configuration or recalibrate promo date enforcement).
- Mention at least two high-priority SKUs with the most exceptions (pick the top SKUs by Total Errors from the Summary, breaking ties by SKU ascending).

## Step 4: Validate before finishing
- Reopen `/root/Promo_Register_Audit.xlsx` and confirm: sheet names are exactly `RawData`, `Formatted Data`, `Summary`; Formatted Data has 12 columns with the exact headers; values in columns 9-12 are concrete numbers/strings, not formulas; Summary headers exact; last row is `Grand Total` / `-`.
- Verify the Error Summary string matches one of the four allowed values for every row.
- Confirm `/root/Promo_Register_Brief.docx` exists, opens, and contains the required content.
- Spot-check 2-3 rows by hand to ensure Price Error and Window Error logic is correct.

## Constraints
- Exact filenames: `/root/Promo_Register_Audit.xlsx` and `/root/Promo_Register_Brief.docx`.
- Exact sheet names and header strings (including capitalization and spacing).
- Do not use spreadsheet formulas for the computed columns; write literal values.

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