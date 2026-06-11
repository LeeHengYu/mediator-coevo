# Task Instruction

Execute the following steps in a single Python script to produce the two deliverables.

## Step 0 – Inspect the source file
```python
import openpyxl
wb_src = openpyxl.load_workbook('/root/Receiving_Log.xlsx')
ws_src = wb_src.active
print('Sheet name:', ws_src.title)
print('Dimensions:', ws_src.dimensions)
print('Max row:', ws_src.max_row, 'Max col:', ws_src.max_column)
for row in ws_src.iter_rows(min_row=1, max_row=min(5, ws_src.max_row), values_only=False):
    print([cell.value for cell in row])
```
Run this first, read the output, then proceed.

## Step 1 – Build `/root/Receiving_Exception_Audit.xlsx`

Use **openpyxl only** (no pandas) to avoid NaN issues. Read every cell from the source workbook as-is (preserving strings, numbers, and None). **Critical**: when writing cells, if a source cell value is `None`, write the **string `None`** only if you are sure the source truly has an empty cell AND the column is one that might contain text. However, the safest approach for the RawData sheet is: copy every cell value exactly as openpyxl reads it (including None for truly empty cells). But be aware of the cross-task warning: if any cell reads as None but should logically be a string like 'N/A', preserve whatever the source file actually contains. Do NOT convert None to 'N/A' unless the source cell is truly empty and context demands it. The key principle: **copy source cells verbatim**.

### Sheet 1: `RawData`
- Copy all rows and columns from the source sheet exactly as they appear (headers + data), preserving original values and types.
- Do not transform, fill, or alter any values.

### Sheet 2: `Formatted Data`
- Copy the first 8 columns from RawData (same row order, same values).
- The first 8 column headers must be exactly: `Receipt ID`, `Item Code`, `Expected Qty`, `Received Qty`, `Storage Class`, `Temp Status`, `Supplier`, `Dock`.
- If the source headers differ in naming, rename them to match these exact names in Formatted Data (but keep RawData headers as-is from source).
- Add columns 9–12 with headers: `Qty Variance`, `Cold Chain Error`, `Total Errors`, `Error Summary`.
- For each data row, compute:
  - `Qty Variance` = 1 if Received Qty != Expected Qty, else 0. (Compare as numbers.)
  - `Cold Chain Error` = 1 if `str(Storage Class).strip().upper()` is in `{'CHILLED', 'FROZEN'}` AND `str(Temp Status).strip().upper()` != `'OK'`, else 0.
  - `Total Errors` = Qty Variance + Cold Chain Error (write as integer).
  - `Error Summary`: build from parts list; if Qty Variance==1 add 'Qty Variance'; if Cold Chain Error==1 add 'Cold Chain Error'; join with ', '. If empty, write 'None'.
- Write all computed values as concrete Python int/str values, not Excel formulas.

### Sheet 3: `Summary`
- Headers: `Item Code`, `Supplier`, `Qty Variance Errors`, `Cold Chain Errors`, `Total Errors`.
- Aggregate from Formatted Data by (Item Code, Supplier) pair.
- Sum Qty Variance, Cold Chain Error, Total Errors for each group.
- Include only groups where Total Errors > 0.
- Sort by Item Code ascending, then Supplier ascending (standard string sort).
- Append a final row: `Grand Total`, `-`, and the dataset-wide sums for the three numeric columns.

Make sure the workbook has exactly three sheets named `RawData`, `Formatted Data`, `Summary` (delete any default sheets if needed). Save to `/root/Receiving_Exception_Audit.xlsx`.

## Step 2 – Build `/root/Receiving_Exception_Brief.docx`

Use `python-docx`. Create a document with a short executive summary (3–6 sentences) that includes:
- Plain-language definition of both checks: Qty Variance flags receipts where received quantity differs from expected; Cold Chain Error flags chilled/frozen items with a non-OK temperature status.
- The computed grand totals for Qty Variance errors, Cold Chain errors, and Total Errors (use the actual numbers from the Summary Grand Total row).
- At least one actionable recommendation (e.g., recount high-variance items at dock, recalibrate cold-chain monitoring).
- Mention at least two specific high-priority Item Codes that appear most frequently in the exceptions (pick the two Item Codes with the highest Total Errors from the Summary sheet).

Save to `/root/Receiving_Exception_Brief.docx`.

## Step 3 – Validate
- Re-open `/root/Receiving_Exception_Audit.xlsx` with openpyxl and print:
  - Sheet names
  - First 3 rows of each sheet
  - Last 2 rows of Summary (to confirm Grand Total row)
  - Row counts per sheet
- Confirm `/root/Receiving_Exception_Brief.docx` exists and print its paragraph texts.

## Important Reminders
- Use openpyxl throughout for Excel (not pandas) to avoid NaN/None conversion issues.
- Write integers (not floats) for all numeric computed columns.
- Worksheet names must be exact: `RawData`, `Formatted Data`, `Summary`.
- Output filenames must be exact.
- Do not skip the validation step.

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