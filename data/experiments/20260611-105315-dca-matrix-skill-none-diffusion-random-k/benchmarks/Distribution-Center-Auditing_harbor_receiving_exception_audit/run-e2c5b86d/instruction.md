# Task Instruction

## Task: Distribution Center Receiving Exception Audit

You must produce two deliverable files from the source workbook `/root/Receiving_Log.xlsx`.

### Step 0: Inspect the source data
1. Read `/root/Receiving_Log.xlsx` to understand its structure: sheet names, column headers, number of rows, and sample values.
2. Pay special attention to the exact column names (they must be mapped to: Receipt ID, Item Code, Expected Qty, Received Qty, Storage Class, Temp Status, Supplier, Dock).
3. Note the data types and any edge cases (e.g., mixed case in Storage Class or Temp Status).

### Step 1: Build `/root/Receiving_Exception_Audit.xlsx`

Use Python with `openpyxl` and `pandas`. Create the workbook with exactly three worksheets named: `RawData`, `Formatted Data`, `Summary`.

#### Sheet 1: `RawData`
- Copy the entire source table from `Receiving_Log.xlsx` exactly as-is (same headers, same values, same row order).

#### Sheet 2: `Formatted Data`
- Same row order as RawData.
- First 8 columns with exactly these headers: `Receipt ID`, `Item Code`, `Expected Qty`, `Received Qty`, `Storage Class`, `Temp Status`, `Supplier`, `Dock`.
- Add 4 computed columns (columns 9-12) with exactly these headers: `Qty Variance`, `Cold Chain Error`, `Total Errors`, `Error Summary`.
- Computation rules (write concrete values, NOT Excel formulas):
  - `Qty Variance` = 1 if `Received Qty` != `Expected Qty`, else 0. (Use numeric int 1 or 0.)
  - `Cold Chain Error` = 1 only when `Storage Class` (case-insensitive) is `CHILLED` or `FROZEN` AND `Temp Status` (case-insensitive) is NOT `OK`. Otherwise 0.
  - `Total Errors` = `Qty Variance` + `Cold Chain Error` (integer).
  - `Error Summary` = exactly one of these strings:
    - `"None"` (when both are 0)
    - `"Qty Variance"` (when only Qty Variance is 1)
    - `"Cold Chain Error"` (when only Cold Chain Error is 1)
    - `"Qty Variance, Cold Chain Error"` (when both are 1)

#### Sheet 3: `Summary`
- Headers exactly: `Item Code`, `Supplier`, `Qty Variance Errors`, `Cold Chain Errors`, `Total Errors`.
- Group the `Formatted Data` rows by `(Item Code, Supplier)`.
- For each group, sum `Qty Variance` → `Qty Variance Errors`, sum `Cold Chain Error` → `Cold Chain Errors`, sum `Total Errors` → `Total Errors`.
- Include ONLY groups where `Total Errors > 0`.
- Sort by `Item Code` ascending, then `Supplier` ascending (standard lexicographic).
- Append a final Grand Total row: `Item Code` = `"Grand Total"`, `Supplier` = `"-"`, and the remaining three columns = sums across the entire dataset (i.e., sum of all Qty Variance, all Cold Chain Error, all Total Errors from Formatted Data, not just the filtered groups — actually these should be the same since groups with 0 total errors contribute 0).

### Step 2: Build `/root/Receiving_Exception_Brief.docx`

Use `python-docx`. Create a Word document with:
- A heading (e.g., "Receiving Exception Audit – Executive Summary").
- A short executive summary paragraph (3-6 sentences) that includes ALL of the following:
  1. A plain-language definition of the Qty Variance check (received quantity differs from expected quantity).
  2. A plain-language definition of the Cold Chain Error check (chilled or frozen items with a temperature status other than OK).
  3. The exact computed totals: total Qty Variance errors, total Cold Chain errors, and total combined errors (use the actual numbers from the data).
  4. At least one actionable recommendation (e.g., recount procedures, supplier corrective action, temperature monitoring improvements).
  5. Mention at least two specific high-priority Item Codes that have the most frequent exceptions (identify these from the Summary data — pick the top 2 Item Codes by Total Errors).

### Step 3: Validation
After creating both files, verify:
1. Open `/root/Receiving_Exception_Audit.xlsx` and confirm:
   - Exactly 3 sheets with names `RawData`, `Formatted Data`, `Summary`.
   - `RawData` has the same number of data rows as the source.
   - `Formatted Data` has 12 columns with exact header names.
   - `Formatted Data` computed columns contain integers (0/1) and correct Error Summary strings.
   - `Summary` has 5 columns with exact header names.
   - `Summary` last row has Item Code = `Grand Total` and Supplier = `-`.
   - `Summary` rows (excluding Grand Total) all have Total Errors > 0.
   - `Summary` is sorted by Item Code then Supplier ascending.
   - Grand Total row sums match the column sums of the summary data rows (and also match the sums from Formatted Data).
2. Open `/root/Receiving_Exception_Brief.docx` and confirm it exists and contains the required content.
3. Print a summary of key statistics (total rows, total qty variance errors, total cold chain errors, total errors, number of summary groups) for verification.

### Important Notes
- Install any needed packages (`pip install openpyxl python-docx pandas`) at the start.
- Filenames and sheet names must be EXACTLY as specified — case-sensitive.
- All computed values must be written as concrete values (int/str), not Excel formulas.
- For the Error Summary column, use exactly the strings specified with exact punctuation: `"Qty Variance, Cold Chain Error"` (note the comma and space).

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