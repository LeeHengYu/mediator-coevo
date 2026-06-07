# Task Instruction

Create two deliverables for a promotional register audit:

1. `/root/Promo_Register_Audit.xlsx` with sheets `RawData`, `Formatted Data`, `Summary`.
2. `/root/Promo_Register_Brief.docx` with a 3-6 sentence executive summary.

Source: `/root/Promo_Price_Check_Source.xlsx`.

## Critical Preliminary Step: Inspect Source

Before writing anything, run a small Python script to:
- Load the source workbook with openpyxl (read_only=False, data_only=True) and iterate cells in the source sheet.
- For each cell, print `(value, type(value).__name__, cell.number_format)`.
- Determine whether date columns (`Promo Start Date`, `Sale Date`, `Promo End Date`) are stored as Excel date serials (cell.value is datetime, with a date number_format) OR as text strings (e.g., '2026-03-01').
- Also note any literal sentinel strings like 'N/A', empty strings, or other non-numeric tokens, and preserve them verbatim.

The previous run failed because dates ended up as `datetime` objects but the validator expected the string '2026-03-01'. This strongly suggests the source stores them as ISO date strings (text cells), not Excel dates. Confirm via inspection. If the source cells are strings, you MUST write strings (do not let pandas auto-parse them into Timestamps). If the source cells are real Excel datetimes, write them as datetimes with the same number_format so they render identically.

## Recommended Implementation

Use openpyxl directly (not pandas `read_excel`, which silently coerces date-like strings) to preserve literals exactly.

Step A: Read source with openpyxl, collecting rows as a list of lists with raw cell values untouched. Preserve strings as strings, numbers as numbers, None as None (but if a cell holds the literal text 'N/A' or similar, keep it as that string).

Step B: Write `RawData` sheet by copying those rows verbatim, cell by cell, into the new workbook. Do not pass through pandas. Do not convert types.

Step C: Build `Formatted Data`:
- First 8 columns: identical values to `RawData` (same types, same row order).
- Add headers in columns 9-12: `Price Error`, `Window Error`, `Total Errors`, `Error Summary`.
- For computations, parse the date values into `datetime.date` objects in memory only (using `datetime.fromisoformat` if strings, or `.date()` if datetimes) for comparison purposes. Do NOT write these parsed values back to the sheet.
- For each data row, compute:
  - `Price Error` = 1 if Register Price != Promo Price else 0 (compare as numbers; be tolerant of float vs int but do not round away real differences).
  - `Window Error` = 1 if Sale Date < Promo Start Date or Sale Date > Promo End Date else 0.
  - `Total Errors` = Price Error + Window Error.
  - `Error Summary` is exactly one of: `None`, `Price Error`, `Window Error`, `Price Error, Window Error`.
- Write Price Error, Window Error, Total Errors as integers (not strings, not formulas). Write Error Summary as a plain string.

Step D: Build `Summary` sheet:
- Headers: `SKU`, `Store ID`, `Price Errors`, `Window Errors`, `Total Errors`.
- Aggregate from Formatted Data grouped by (SKU, Store ID).
- Include only groups where Total Errors > 0.
- Sort by SKU ascending, then Store ID ascending.
- Append a final row: SKU='Grand Total', Store ID='-', then dataset totals for Price Errors, Window Errors, Total Errors (sums across the included groups, which equals dataset totals since groups with 0 errors contribute 0).

Step E: Save the workbook to `/root/Promo_Register_Audit.xlsx`.

Step F: Create `/root/Promo_Register_Brief.docx` using python-docx with 3-6 sentences covering:
- Plain-language definition of Price Error (register charged a price different from the promotional price) and Window Error (sale occurred outside the promo's start/end window).
- The computed totals: total Price Errors, total Window Errors, total Total Errors.
- At least one actionable recommendation (e.g., re-sync register price tables, audit promo calendar configuration).
- Name at least two specific SKUs that had the most exceptions (pick top 2 by total errors across stores from the Summary).

## Validation Before Finishing

Run a verification script that:
1. Opens `/root/Promo_Register_Audit.xlsx` with openpyxl and the source workbook side by side. For every cell in `RawData`, assert `target.value == source.value` AND `type(target.value) == type(source.value)`. This catches the date-as-datetime vs date-as-string bug from the previous run, and also catches 'N/A' → None coercion.
2. Verify `Formatted Data` first 8 columns match `RawData` exactly (same types and values).
3. Verify column 9-12 headers and that Price Error/Window Error/Total Errors values are ints, Error Summary is one of the four allowed strings.
4. Verify Summary sheet headers, sort order, exclusion of zero-error groups, and Grand Total row.
5. Verify the .docx exists, opens, and contains the required pieces (both definitions, three totals, a recommendation, two SKU mentions).

If any assertion fails, fix and re-run before declaring done.

## Key Pitfalls to Avoid
- Do NOT use `pandas.read_excel` for the copy step; it coerces date-like strings into Timestamps and 'N/A' into NaN/None.
- Do NOT write `datetime` objects when the source cells contain ISO date strings.
- Do NOT normalize, reformat, or strip any source literals.
- Do NOT use Excel formulas for computed columns; write concrete values.
- Do NOT change sheet names, file names, or column header text.

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