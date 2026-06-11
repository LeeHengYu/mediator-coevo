# Task Instruction

Execute the following steps in a single Python script to produce both deliverables.

## Step 1: Read the source data
```python
import openpyxl
from openpyxl import Workbook
from docx import Document
import pandas as pd

# Read source workbook
df = pd.read_excel('/root/Receiving_Log.xlsx')
```

## Step 2: Build `/root/Receiving_Exception_Audit.xlsx`

### Sheet 1 – `RawData`
- Write the source data exactly as-is (same columns, same row order) into a sheet named `RawData`.

### Sheet 2 – `Formatted Data`
- Start from the same data and row order.
- Keep the first 8 columns with these exact headers: `Receipt ID`, `Item Code`, `Expected Qty`, `Received Qty`, `Storage Class`, `Temp Status`, `Supplier`, `Dock`.
- Add four computed columns (write concrete values, NOT Excel formulas):
  - **Qty Variance** (int): 1 if `Received Qty` != `Expected Qty`, else 0.
  - **Cold Chain Error** (int): 1 only when `Storage Class` (case-insensitive) is `CHILLED` or `FROZEN` AND `Temp Status` (case-insensitive) is NOT `OK`. Otherwise 0.
  - **Total Errors** (int): `Qty Variance + Cold Chain Error`.
  - **Error Summary** (str): exactly one of `None`, `Qty Variance`, `Cold Chain Error`, or `Qty Variance, Cold Chain Error` — determined by which flags are 1.
- Ensure all numeric columns (`Expected Qty`, `Received Qty`, `Qty Variance`, `Cold Chain Error`, `Total Errors`) are written as Python ints, not floats.

### Sheet 3 – `Summary`
- Aggregate from the Formatted Data by `(Item Code, Supplier)`.
- Columns: `Item Code`, `Supplier`, `Qty Variance Errors`, `Cold Chain Errors`, `Total Errors`.
  - `Qty Variance Errors` = sum of `Qty Variance` for the group.
  - `Cold Chain Errors` = sum of `Cold Chain Error` for the group.
  - `Total Errors` = sum of `Total Errors` for the group.
- **Filter**: include only groups where `Total Errors > 0`.
- **Sort**: by `Item Code` ascending, then `Supplier` ascending.
- **Grand Total row**: `Item Code` = `Grand Total`, `Supplier` = `-`, remaining columns = dataset-wide totals (sum over ALL rows in Formatted Data, not just filtered groups — though they should be the same since zero-error groups contribute nothing).
- Write all numeric values as ints.

Use `openpyxl` to write the workbook so you have full control over sheet names and cell values. Do NOT use `pd.ExcelWriter` with engine='openpyxl' unless you explicitly set sheet names. Verify sheet names are exactly `RawData`, `Formatted Data`, `Summary`.

## Step 3: Build `/root/Receiving_Exception_Brief.docx`

Using `python-docx`, create a Word document with a short executive summary (3–6 sentences) that includes:
1. A plain-language definition of both checks: Qty Variance (received quantity differs from expected) and Cold Chain Error (chilled/frozen item with a temperature status other than OK).
2. The exact computed totals: total Qty Variance errors, total Cold Chain errors, and overall Total Errors (use the Grand Total values).
3. At least one actionable recommendation (e.g., recount procedures, cold-chain protocol review).
4. Mention at least two specific high-priority item codes that appear most frequently in the exceptions (pick the top 2 item codes by Total Errors from the Summary sheet).

Save as `/root/Receiving_Exception_Brief.docx`.

## Step 4: Validate
- Re-open `/root/Receiving_Exception_Audit.xlsx` with openpyxl and confirm:
  - Sheet names are exactly `['RawData', 'Formatted Data', 'Summary']`.
  - `Formatted Data` has 12 columns with correct headers.
  - `Summary` last row has `Grand Total` in column A.
  - Spot-check a few Qty Variance and Cold Chain Error values.
- Confirm `/root/Receiving_Exception_Brief.docx` exists and is non-empty.
- Print confirmation messages for each check.

## Important Notes
- Use case-insensitive comparisons: `str(val).strip().upper()` for Storage Class and Temp Status.
- Handle any NaN/blank values in Storage Class or Temp Status by treating them as non-matching (Cold Chain Error = 0 for blanks).
- Cast numeric values to int before writing to avoid float artifacts (e.g., `1.0` instead of `1`).
- Do not rename or reorder the first 8 columns — use whatever headers exist in the source, but ensure the output headers match the spec exactly. If the source headers differ, rename them in the output.

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