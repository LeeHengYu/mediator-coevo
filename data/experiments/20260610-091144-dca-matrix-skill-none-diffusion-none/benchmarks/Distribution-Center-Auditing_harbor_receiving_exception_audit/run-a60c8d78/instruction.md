# Task Instruction

Write and execute a Python script that:

1. Reads `/root/Receiving_Log.xlsx` into a pandas DataFrame.
2. Creates `/root/Receiving_Exception_Audit.xlsx` with exactly three worksheets:

   **Sheet `RawData`:**
   - Copy the source data exactly as-is (preserve all columns, rows, and order).

   **Sheet `Formatted Data`:**
   - Same row order as RawData.
   - First 8 columns exactly: `Receipt ID`, `Item Code`, `Expected Qty`, `Received Qty`, `Storage Class`, `Temp Status`, `Supplier`, `Dock`.
   - Add 4 new columns (9-12):
     - `Qty Variance`: integer 1 if `Received Qty != Expected Qty`, else 0.
     - `Cold Chain Error`: integer 1 only when `Storage Class` (case-insensitive) is `CHILLED` or `FROZEN` **and** `Temp Status` (case-insensitive) is **not** `OK`. Otherwise 0.
     - `Total Errors`: integer = `Qty Variance + Cold Chain Error`.
     - `Error Summary`: exactly one of `None`, `Qty Variance`, `Cold Chain Error`, or `Qty Variance, Cold Chain Error` (use these exact strings, with the comma-space separator).
   - All four added columns must be written as concrete values (int for numeric, str for Error Summary), not Excel formulas.

   **Sheet `Summary`:**
   - Headers exactly: `Item Code`, `Supplier`, `Qty Variance Errors`, `Cold Chain Errors`, `Total Errors`.
   - Group `Formatted Data` by (`Item Code`, `Supplier`), summing the three error columns.
   - Include only groups where `Total Errors > 0`.
   - Sort by `Item Code` ascending then `Supplier` ascending.
   - Append a final row: `Item Code` = `Grand Total`, `Supplier` = `-`, remaining columns = dataset-wide totals of the three error columns.
   - Ensure all numeric columns are int (not float).

3. Creates `/root/Receiving_Exception_Brief.docx`:
   - A short executive summary (3-6 sentences).
   - Include a plain-language definition of both checks (Qty Variance: received quantity differs from expected; Cold Chain Error: chilled/frozen item with a non-OK temperature status).
   - State the computed totals for Qty Variance errors, Cold Chain errors, and Total Errors.
   - Identify at least two high-priority item codes with the most frequent exceptions (derive from the Summary data).
   - Include at least one actionable recommendation.

4. After generating both files, run verification:
   - Print the shape and first few rows of each sheet in the Excel file.
   - Print the column names of `Formatted Data` to confirm all 12 headers.
   - Print the last row of `Summary` to confirm Grand Total row.
   - Print the full text content of the Word document.
   - Confirm both output files exist.

Implementation notes:
- Use `openpyxl` engine for Excel writing.
- Use `python-docx` for Word.
- When comparing strings case-insensitively, use `.str.strip().str.upper()` to handle whitespace.
- Cast `Expected Qty` and `Received Qty` to numeric (coerce errors) before comparison.
- Write each sheet using `index=False` to avoid extra index columns.
- For the Grand Total row, compute sums from the full `Formatted Data` (not just the filtered summary groups).

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