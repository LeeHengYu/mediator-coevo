# Task Instruction

## Task: Distribution Center Receiving Exception Audit

You must create two deliverable files from the source workbook `/root/Receiving_Log.xlsx`.

### Step 1: Inspect the source data
- Open and read `/root/Receiving_Log.xlsx` to understand its structure: column names, number of rows, data types, and sample values.
- Pay special attention to the exact column names (they must be mapped to: Receipt ID, Item Code, Expected Qty, Received Qty, Storage Class, Temp Status, Supplier, Dock).
- Note the exact casing and values in `Storage Class` (look for CHILLED, FROZEN, etc.) and `Temp Status` (look for OK vs other values).

### Step 2: Build `/root/Receiving_Exception_Audit.xlsx`

Use Python with `openpyxl` (and `pandas` for data manipulation if helpful). The workbook must have exactly three worksheets named `RawData`, `Formatted Data`, and `Summary`.

#### Sheet 1: `RawData`
- Copy the entire source table from `Receiving_Log.xlsx` exactly as-is (same headers, same values, same row order).

#### Sheet 2: `Formatted Data`
- Same row order as RawData.
- First 8 columns must be exactly: `Receipt ID`, `Item Code`, `Expected Qty`, `Received Qty`, `Storage Class`, `Temp Status`, `Supplier`, `Dock`.
- Add 4 new columns (columns 9–12) with these exact headers:
  - `Qty Variance`
  - `Cold Chain Error`
  - `Total Errors`
  - `Error Summary`
- Compute values as concrete numbers/text (NOT Excel formulas):
  - `Qty Variance` = 1 if `Received Qty` != `Expected Qty`, else 0
  - `Cold Chain Error` = 1 if `Storage Class` (case-insensitive) is in {"CHILLED", "FROZEN"} AND `Temp Status` (case-insensitive) is NOT "OK"; else 0
  - `Total Errors` = `Qty Variance` + `Cold Chain Error`
  - `Error Summary` = one of exactly these strings:
    - `"None"` (if both are 0)
    - `"Qty Variance"` (if only qty variance)
    - `"Cold Chain Error"` (if only cold chain)
    - `"Qty Variance, Cold Chain Error"` (if both are 1)

#### Sheet 3: `Summary`
- Headers exactly: `Item Code`, `Supplier`, `Qty Variance Errors`, `Cold Chain Errors`, `Total Errors`
- Group the `Formatted Data` rows by `(Item Code, Supplier)` pair.
- Include ONLY groups where the sum of `Total Errors` > 0.
- Sort by `Item Code` ascending, then `Supplier` ascending.
- Append a final row: `Item Code` = `"Grand Total"`, `Supplier` = `"-"`, and the remaining three columns = the sum totals across the entire dataset (sum of all Qty Variance, all Cold Chain Error, all Total Errors from Formatted Data — or equivalently from the summary rows above).

### Step 3: Build `/root/Receiving_Exception_Brief.docx`

Use `python-docx`. Create a Word document with an executive summary paragraph (3–6 sentences) that includes ALL of:
1. A plain-language definition of the Qty Variance check (received quantity differs from expected).
2. A plain-language definition of the Cold Chain Error check (chilled/frozen items with a non-OK temperature status).
3. The exact computed totals: total Qty Variance errors, total Cold Chain errors, and overall Total Errors across the dataset.
4. At least one actionable recommendation (e.g., retraining dock staff, supplier audits, temperature monitoring improvements).
5. Mention at least two specific high-priority Item Codes that have the most frequent exceptions (identify these from the Summary data — pick the top 2 item codes by total errors).

### Step 4: Validate
- Re-open `/root/Receiving_Exception_Audit.xlsx` and verify:
  - Exactly 3 sheets with exact names: `RawData`, `Formatted Data`, `Summary`
  - `RawData` row count matches source
  - `Formatted Data` has 12 columns with correct headers
  - `Formatted Data` computed columns have integer values (0 or 1 for Qty Variance and Cold Chain Error)
  - `Summary` last row has `Item Code` = `"Grand Total"` and `Supplier` = `"-"`
  - Grand Total row sums match the sums from Formatted Data
- Re-open `/root/Receiving_Exception_Brief.docx` and print its text to confirm it contains the required elements.
- Print confirmation of all checks passing.

### Important Notes
- Use `int` type for all numeric error columns (0 and 1, not booleans or floats).
- Write concrete values, not Excel formulas.
- Ensure worksheet names are exactly as specified (note the space in `Formatted Data`).
- Ensure the Grand Total row's numeric columns equal the column sums from the Summary table above it (which should also equal the column sums from the full Formatted Data sheet).

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