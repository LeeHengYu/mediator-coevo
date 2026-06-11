# Task Instruction

## Task: Distribution Center Receiving Exception Audit

You must produce two files from the source workbook `/root/Receiving_Log.xlsx`:
1. `/root/Receiving_Exception_Audit.xlsx`
2. `/root/Receiving_Exception_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect the source data
- Open and read `/root/Receiving_Log.xlsx` using `openpyxl` (or pandas). Print the sheet names, column headers, row count, and the first 5 rows so you understand the exact column names and data types before proceeding.
- Identify which columns correspond to: Receipt ID, Item Code, Expected Qty, Received Qty, Storage Class, Temp Status, Supplier, Dock. Note the exact original header strings.

#### Step 1: Build the output Excel workbook

Use `openpyxl` to create `/root/Receiving_Exception_Audit.xlsx` with exactly three worksheets named `RawData`, `Formatted Data`, and `Summary`.

##### Sheet 1: `RawData`
- Copy the entire source table (headers + all data rows) verbatim from `Receiving_Log.xlsx`. Preserve the original column order, header names, and all cell values exactly.

##### Sheet 2: `Formatted Data`
- Same row order as `RawData`.
- The first 8 columns must have exactly these headers (regardless of original header names): `Receipt ID`, `Item Code`, `Expected Qty`, `Received Qty`, `Storage Class`, `Temp Status`, `Supplier`, `Dock`.
- Map source columns to these 8 headers by semantic meaning (inspect the source headers carefully).
- Add four new columns (columns 9–12) with exactly these headers: `Qty Variance`, `Cold Chain Error`, `Total Errors`, `Error Summary`.
- Compute values as concrete numbers/strings (NOT Excel formulas):
  - `Qty Variance` = 1 if `Received Qty` != `Expected Qty`, else 0. Compare numerically.
  - `Cold Chain Error` = 1 if `Storage Class` (case-insensitive) is in {`CHILLED`, `FROZEN`} AND `Temp Status` (case-insensitive) is NOT `OK`. Otherwise 0.
  - `Total Errors` = `Qty Variance` + `Cold Chain Error` (integer).
  - `Error Summary` = exactly one of these four strings:
    - `None` (if both are 0)
    - `Qty Variance` (if only qty variance is 1)
    - `Cold Chain Error` (if only cold chain error is 1)
    - `Qty Variance, Cold Chain Error` (if both are 1)

##### Sheet 3: `Summary`
- Headers (exactly): `Item Code`, `Supplier`, `Qty Variance Errors`, `Cold Chain Errors`, `Total Errors`.
- Group the `Formatted Data` rows by `(Item Code, Supplier)`. For each group, sum `Qty Variance` → `Qty Variance Errors`, sum `Cold Chain Error` → `Cold Chain Errors`, sum `Total Errors` → `Total Errors`.
- Include only groups where `Total Errors > 0`.
- Sort by `Item Code` ascending (alphabetical/lexicographic), then `Supplier` ascending.
- After all data rows, append one final row: `Grand Total`, `-`, and the dataset-wide sums for the three numeric columns.

Save the workbook. Make sure there is no default `Sheet` worksheet — delete it if openpyxl creates one.

#### Step 2: Build the Word document

Use `python-docx` to create `/root/Receiving_Exception_Brief.docx`.
- Write a short executive summary (3–6 sentences) that includes:
  1. A plain-language definition of both checks: explain what Qty Variance means (received quantity differs from expected) and what Cold Chain Error means (a chilled/frozen item had a temperature status other than OK).
  2. The computed totals: total Qty Variance errors, total Cold Chain errors, and total combined errors (use the Grand Total row values).
  3. At least one actionable recommendation (e.g., increase spot-checks for specific suppliers or item codes).
  4. Mention at least two specific high-priority item codes that had the most frequent exceptions. To determine these, look at the Summary sheet data and pick the two Item Codes with the highest Total Errors (sum across all suppliers for that item code).

#### Step 3: Validate outputs
- Re-open `/root/Receiving_Exception_Audit.xlsx` and verify:
  - Exactly 3 sheets with names `RawData`, `Formatted Data`, `Summary`.
  - `Formatted Data` has 12 columns with the exact headers specified.
  - `Summary` last row has `Item Code` = `Grand Total`.
  - Print the Summary sheet contents to confirm correctness.
- Re-open `/root/Receiving_Exception_Brief.docx` and print its text to confirm it contains the required elements.

### Important
- Install any needed packages: `pip install openpyxl python-docx pandas` at the start.
- Write all computed values as literal numbers/strings, not Excel formulas.
- Filenames and sheet names must match exactly (case-sensitive, spacing-sensitive).
- Do not leave any extra default sheets in the Excel workbook.

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