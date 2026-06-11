# Task Instruction

## Task: Distribution Center Receiving Exception Audit

You must create two deliverable files from the source workbook `/root/Receiving_Log.xlsx`.

### Step 1: Inspect the source data
- Open and read `/root/Receiving_Log.xlsx` to understand the schema, column names, row count, and data types.
- Print the first 10 rows and all column headers.
- Check the exact column names (they should include: Receipt ID, Item Code, Expected Qty, Received Qty, Storage Class, Temp Status, Supplier, Dock — but verify exact spelling/casing).
- Note the total number of data rows.

### Step 2: Build `/root/Receiving_Exception_Audit.xlsx` with exactly 3 worksheets

Use `openpyxl` (install if needed: `pip install openpyxl`) and `pandas`.

#### Sheet 1: `RawData`
- Copy the entire source table exactly as-is (same headers, same row order, same values).

#### Sheet 2: `Formatted Data`
- Start with the same data and row order as RawData.
- Keep the first 8 columns exactly as: Receipt ID, Item Code, Expected Qty, Received Qty, Storage Class, Temp Status, Supplier, Dock.
- Add 4 new computed columns (columns 9–12) with these EXACT headers:
  - `Qty Variance`
  - `Cold Chain Error`
  - `Total Errors`
  - `Error Summary`

- Computation rules (use concrete values, NOT Excel formulas):
  - `Qty Variance` = 1 if `Received Qty` != `Expected Qty`, else 0
  - `Cold Chain Error` = 1 if `Storage Class` (case-insensitive, stripped) is in {"CHILLED", "FROZEN"} AND `Temp Status` (case-insensitive, stripped) is NOT "OK"; else 0
  - `Total Errors` = `Qty Variance` + `Cold Chain Error`
  - `Error Summary` = exactly one of these four strings:
    - `"None"` (if both are 0)
    - `"Qty Variance"` (if only Qty Variance is 1)
    - `"Cold Chain Error"` (if only Cold Chain Error is 1)
    - `"Qty Variance, Cold Chain Error"` (if both are 1)

- IMPORTANT: Write integer values (Python int, not float) for the numeric columns. Write plain strings for Error Summary. Do NOT write Excel formulas.

#### Sheet 3: `Summary`
- Aggregate from the Formatted Data by (Item Code, Supplier).
- For each group, sum: `Qty Variance Errors`, `Cold Chain Errors`, `Total Errors`.
- Include ONLY groups where Total Errors > 0.
- Sort by Item Code ascending, then Supplier ascending.
- Headers must be exactly: `Item Code`, `Supplier`, `Qty Variance Errors`, `Cold Chain Errors`, `Total Errors`
- Append a final row: Item Code = `Grand Total`, Supplier = `-`, and the remaining 3 columns = dataset-wide totals (sum of all Qty Variance, sum of all Cold Chain Error, sum of all Total Errors from the full Formatted Data sheet, not just the filtered groups — but since groups with 0 total errors contribute 0, the sums should be the same).

### Step 3: Build `/root/Receiving_Exception_Brief.docx`

Install python-docx if needed: `pip install python-docx`

Create a Word document with:
- A heading like "Receiving Exception Brief" or "Executive Summary"
- A short executive summary paragraph (3–6 sentences) that includes ALL of the following:
  1. A plain-language definition of the Qty Variance check (flags when received quantity differs from expected quantity).
  2. A plain-language definition of the Cold Chain Error check (flags when chilled or frozen items have a temperature status other than OK).
  3. The computed totals: total Qty Variance errors, total Cold Chain errors, and total combined errors (use the actual numbers from the data).
  4. At least one actionable recommendation (e.g., retraining dock staff, recalibrating temperature monitors, supplier quality review).
  5. Mention at least two specific high-priority Item Codes that have the most frequent exceptions (find the top 2 Item Codes by Total Errors from the Summary data).

### Step 4: Validation
- Re-read the generated Excel file and verify:
  - Sheet names are exactly: `RawData`, `Formatted Data`, `Summary`
  - `RawData` has the same number of rows as the source
  - `Formatted Data` has 12 columns with the correct headers
  - The computed columns contain integers (not floats like 1.0) and correct string values
  - `Summary` has exactly 5 columns with the correct headers
  - `Summary` last row has Item Code = `Grand Total` and Supplier = `-`
  - `Summary` rows (excluding Grand Total) are sorted by Item Code asc, then Supplier asc
  - `Summary` only includes groups with Total Errors > 0
- Re-read the Word file and verify it contains the required elements.
- Print confirmation of all checks.

### Technical Notes
- When writing with pandas to Excel using openpyxl engine, be careful: use `index=False` to avoid writing the DataFrame index.
- For the Formatted Data numeric columns, explicitly cast to Python `int` before writing to avoid float issues.
- When comparing Storage Class and Temp Status, use `.str.strip().str.upper()` to handle case and whitespace.
- Make sure the Grand Total row values are Python ints, not numpy int64 (convert with `int()`).
- If the source workbook has a specific sheet name, read from that sheet. If it has only one sheet, read the first one.

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