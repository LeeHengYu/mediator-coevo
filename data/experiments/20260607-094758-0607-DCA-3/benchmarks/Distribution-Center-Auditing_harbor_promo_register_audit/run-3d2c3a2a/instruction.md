# Task Instruction

Build two deliverables from `/root/Promo_Price_Check_Source.xlsx`: an Excel audit workbook `/root/Promo_Register_Audit.xlsx` and a Word brief `/root/Promo_Register_Brief.docx`.

STEP 1 — Inspect the source.
- Open `/root/Promo_Price_Check_Source.xlsx` with pandas/openpyxl. Confirm the sheet name and verify the exact column headers and order. Expected columns: Promo ID, SKU, Promo Price, Register Price, Promo Start Date, Sale Date, Promo End Date, Store ID. Note dtypes (especially date columns) and row count.

STEP 2 — Build `/root/Promo_Register_Audit.xlsx` with exactly three sheets, in this order: `RawData`, `Formatted Data`, `Summary`.

Sheet `RawData`:
- Write the source table verbatim. Preserve column order, header names, row order, and values. Do not reformat dates or numbers.

Sheet `Formatted Data`:
- Same row order as `RawData`.
- Columns 1–8 identical to the source 8 columns listed above (same headers, same values).
- Append columns 9–12 with headers exactly: `Price Error`, `Window Error`, `Total Errors`, `Error Summary`.
- Compute per row using concrete values (no formulas):
  - Parse the three date columns as calendar dates (use pandas.to_datetime; normalize to date only to avoid time-of-day comparison artifacts).
  - `Price Error` = 1 if Register Price != Promo Price else 0. Use numeric comparison; if values are floats, compare with a tight tolerance (e.g., round to 2 decimals) only if needed — otherwise compare directly.
  - `Window Error` = 1 if Sale Date < Promo Start Date OR Sale Date > Promo End Date else 0.
  - `Total Errors` = Price Error + Window Error (integer 0/1/2).
  - `Error Summary` exactly one of: `None`, `Price Error`, `Window Error`, `Price Error, Window Error` (match spelling, capitalization, and the comma+space separator).
- Write integers as integers (not floats) for the three numeric error columns.

Sheet `Summary`:
- Headers exactly: `SKU`, `Store ID`, `Price Errors`, `Window Errors`, `Total Errors`.
- Group `Formatted Data` by (SKU, Store ID); sum Price Error → Price Errors, Window Error → Window Errors, Total Errors → Total Errors.
- Keep only groups where Total Errors > 0.
- Sort by SKU ascending, then Store ID ascending.
- Append final row: SKU=`Grand Total`, Store ID=`-`, Price Errors=sum, Window Errors=sum, Total Errors=sum (totals computed over the included groups — which equals dataset totals since excluded groups contribute 0).

Use openpyxl or pandas ExcelWriter. Ensure sheet names are exactly `RawData`, `Formatted Data`, `Summary` (note the space in `Formatted Data`).

STEP 3 — Build `/root/Promo_Register_Brief.docx` with python-docx.
- 3–6 sentences in an executive-summary style.
- Include: (a) plain-language definitions of Price Error (register price differed from promo price) and Window Error (sale occurred outside the promo start–end window); (b) the dataset totals for Price Errors, Window Errors, and Total Errors (use the numbers computed above); (c) at least one actionable recommendation (e.g., reconcile register price files before promo start, audit POS date settings); (d) name at least two high-priority SKUs — pick the SKUs with the highest Total Errors from the Summary sheet (break ties by Price Errors then Window Errors then SKU).

STEP 4 — Validate before finishing.
- Reopen `/root/Promo_Register_Audit.xlsx` and assert: sheet names == [`RawData`, `Formatted Data`, `Summary`]; Formatted Data has exactly 12 columns with the exact headers in order; Error Summary values are within the allowed set; Total Errors == Price Error + Window Error on every row; Summary excludes zero-total groups; Summary is sorted; final row is Grand Total / `-` with sums matching column totals of Formatted Data error columns.
- Reopen the docx and confirm it contains the two definitions, the three totals as numbers, a recommendation, and at least two SKU identifiers.
- Report any validation failure and fix before completion.

Constraints: keep filenames and sheet names exact; do not use spreadsheet formulas for the computed columns; do not modify or weaken these checks.

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