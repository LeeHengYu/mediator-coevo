# Task Instruction

Execute the following steps in a single Python script to produce the two deliverables.

## Step 0 – Inspect the source file
```python
import pandas as pd
df = pd.read_excel('/root/Receiving_Log.xlsx')
print(df.columns.tolist())
print(df.head())
print(df.dtypes)
print(len(df))
```
Confirm the column names. They should include at least: Receipt ID, Item Code, Expected Qty, Received Qty, Storage Class, Temp Status, Supplier, Dock. If names differ slightly (whitespace, casing), note the exact names and map them in subsequent steps.

## Step 1 – Build the output workbook `/root/Receiving_Exception_Audit.xlsx`

### Sheet 1: `RawData`
- Copy the source DataFrame exactly (same columns, same order, same values) into a sheet named `RawData`.

### Sheet 2: `Formatted Data`
- Start from the same DataFrame (same row order).
- Keep only the first 8 columns in this exact order and with these exact header strings:
  1. Receipt ID
  2. Item Code
  3. Expected Qty
  4. Received Qty
  5. Storage Class
  6. Temp Status
  7. Supplier
  8. Dock
- If the source column names differ, rename them to match exactly.
- Add four computed columns (concrete values, NOT Excel formulas):
  9. **Qty Variance**: 1 if Received Qty ≠ Expected Qty, else 0. Use int type.
  10. **Cold Chain Error**: 1 only when Storage Class (case-insensitive) is in {"CHILLED", "FROZEN"} AND Temp Status (case-insensitive) is NOT "OK". Otherwise 0. Use int type.
  11. **Total Errors**: Qty Variance + Cold Chain Error. Use int type.
  12. **Error Summary**: Exactly one of these four strings:
      - `None` (the string, not Python None)
      - `Qty Variance`
      - `Cold Chain Error`
      - `Qty Variance, Cold Chain Error`

  Build Error Summary with logic:
  ```python
  def error_summary(row):
      parts = []
      if row['Qty Variance'] == 1:
          parts.append('Qty Variance')
      if row['Cold Chain Error'] == 1:
          parts.append('Cold Chain Error')
      return ', '.join(parts) if parts else 'None'
  ```

### Sheet 3: `Summary`
- Aggregate from the Formatted Data by (Item Code, Supplier).
- Compute per-group sums: Qty Variance Errors (sum of Qty Variance), Cold Chain Errors (sum of Cold Chain Error), Total Errors (sum of Total Errors).
- Filter: keep only groups where Total Errors > 0.
- Sort by Item Code ascending, then Supplier ascending.
- Headers must be exactly: Item Code, Supplier, Qty Variance Errors, Cold Chain Errors, Total Errors.
- Append a final row: Item Code = 'Grand Total', Supplier = '-', and the remaining three columns = dataset-wide totals (summed from the kept rows).
- Write as sheet `Summary`.

Use `pd.ExcelWriter('/root/Receiving_Exception_Audit.xlsx', engine='openpyxl')` and write all three sheets. Set `index=False` for every sheet.

## Step 2 – Build the Word document `/root/Receiving_Exception_Brief.docx`

Using `python-docx`:
- Add a heading: "Receiving Exception Brief"
- Write 3–6 sentences covering:
  1. Plain-language definition of Qty Variance check: flags receipts where the received quantity does not match the expected quantity.
  2. Plain-language definition of Cold Chain Error check: flags receipts of chilled or frozen items where the temperature status was not OK.
  3. State the computed totals: total Qty Variance errors = X, total Cold Chain errors = Y, total combined errors = Z (use actual numbers from the data).
  4. Identify at least two Item Codes with the highest number of total exceptions.
  5. Include at least one actionable recommendation (e.g., investigate supplier compliance, recalibrate dock thermometers, tighten receiving SOP).
- Save to `/root/Receiving_Exception_Brief.docx`.

## Step 3 – Validate
- Re-read `/root/Receiving_Exception_Audit.xlsx` and confirm:
  - Sheet names are exactly ['RawData', 'Formatted Data', 'Summary'].
  - `Formatted Data` has 12 columns with the exact headers listed above.
  - `Summary` last row has Item Code == 'Grand Total'.
  - All Qty Variance, Cold Chain Error, Total Errors values are int (0 or positive).
  - Error Summary values are only from the four allowed strings.
- Re-read `/root/Receiving_Exception_Brief.docx` and print its text to confirm content.
- Print 'ALL VALIDATIONS PASSED' if everything checks out.

Execute all steps in order. If the source column names don't match exactly, map them carefully before proceeding. Do not skip the validation step.

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