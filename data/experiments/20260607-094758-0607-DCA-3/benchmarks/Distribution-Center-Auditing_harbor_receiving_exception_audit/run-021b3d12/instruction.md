# Task Instruction

Create two deliverables for a distribution-center receiving exception audit.

## Source
- Read `/root/Receiving_Log.xlsx` (single source workbook).

## Deliverable 1: `/root/Receiving_Exception_Audit.xlsx`

Must have exactly three worksheets in this order: `RawData`, `Formatted Data`, `Summary`.

### Step 0: Inspect the source first
Before writing anything, open the source workbook with openpyxl (values_only or cell-by-cell) and inspect:
- Column headers (exact order, exact spelling).
- Each cell's raw value AND type (str, int, float, datetime, None).
- Look specifically for string literals like 'N/A', 'NA', '-', blanks, and any date-like strings vs datetime objects.
Record findings briefly before proceeding.

### Sheet 1: `RawData`
- Copy the source table EXACTLY. Same headers, same row order, same values, same types as they appear when read by openpyxl.
- CRITICAL: Preserve string literals verbatim. If the source cell contains the string 'N/A', write the string 'N/A' (not None, NaN, or empty).
- CRITICAL: If a source cell is a string like '2026-03-01', write that string, not a datetime. If a source cell is a datetime, write a datetime. Match the source's native type.
- Prefer openpyxl direct cell read/write over pandas to avoid type coercion. If using pandas, use `dtype=object`, `keep_default_na=False`, and `na_filter=False` when reading, and ensure datetimes are not auto-converted.

### Sheet 2: `Formatted Data`
- First 8 columns must be identical to RawData columns 1-8 with these exact headers and order:
  1. Receipt ID
  2. Item Code
  3. Expected Qty
  4. Received Qty
  5. Storage Class
  6. Temp Status
  7. Supplier
  8. Dock
- Preserve the same row order as RawData.
- Append four new columns (9-12) with exact headers:
  9. Qty Variance
  10. Cold Chain Error
  11. Total Errors
  12. Error Summary
- Compute concrete numeric/text values (NOT formulas):
  - `Qty Variance` = 1 if `Received Qty` != `Expected Qty`, else 0.
  - `Cold Chain Error` = 1 only if `Storage Class` (case-insensitive) is 'CHILLED' or 'FROZEN' AND `Temp Status` (case-insensitive) is not 'OK'. Else 0.
  - `Total Errors` = `Qty Variance` + `Cold Chain Error` (integer).
  - `Error Summary` = exactly one of: 'None', 'Qty Variance', 'Cold Chain Error', 'Qty Variance, Cold Chain Error' based on which flags are 1.

### Sheet 3: `Summary`
Exact headers in this order:
1. Item Code
2. Supplier
3. Qty Variance Errors
4. Cold Chain Errors
5. Total Errors

Rules:
- Aggregate from `Formatted Data` grouped by (Item Code, Supplier).
- For each group sum `Qty Variance`, `Cold Chain Error`, and `Total Errors`.
- Include only groups whose summed `Total Errors > 0`.
- Sort by `Item Code` ascending, then `Supplier` ascending.
- Append a final row:
  - Item Code = 'Grand Total'
  - Supplier = '-'
  - Qty Variance Errors = dataset total of Qty Variance
  - Cold Chain Errors = dataset total of Cold Chain Error
  - Total Errors = dataset total of Total Errors
  (Dataset totals = sum across all Formatted Data rows, not only included groups; but since excluded groups have 0 totals these are equal.)

## Deliverable 2: `/root/Receiving_Exception_Brief.docx`

Write a 3-6 sentence executive summary that includes ALL of:
- A plain-language definition of `Qty Variance` (received quantity differs from expected quantity).
- A plain-language definition of `Cold Chain Error` (CHILLED or FROZEN items received with a non-OK temperature status).
- The computed numeric totals for: Qty Variance errors, Cold Chain errors, and Total Errors (use the exact integers from your computation).
- At least one actionable recommendation.
- At least TWO specific high-priority Item Codes that have the most exceptions (pick the top 2 by Total Errors from the Summary sheet).

## Validation Checklist (run before finishing)
1. Re-open `/root/Receiving_Exception_Audit.xlsx` with openpyxl and verify:
   - Sheet names are exactly ['RawData', 'Formatted Data', 'Summary'].
   - RawData row/column count and values match source exactly; spot-check any 'N/A' literals and date cells preserved with original type.
   - Formatted Data has 12 columns with exact headers; row count equals RawData row count; spot-check the four computed columns on 2-3 rows.
   - Summary has exact headers, only groups with Total Errors > 0, sorted correctly, and a final Grand Total row whose sums match the column sums of Formatted Data.
2. Re-open the docx and verify the totals (as integers) and at least two item codes appear in the text.

## Constraints
- Exact filenames and sheet names.
- Preserve source literals (especially 'N/A') and source types (especially date strings vs datetimes) in RawData and the first 8 columns of Formatted Data.
- Write concrete values, not formulas, in computed columns.
- Do not bypass or modify validators.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=medium, tags=[excel, openpyxl, docx, audit, warehouse].
Verifier config: timeout_sec=900.0.