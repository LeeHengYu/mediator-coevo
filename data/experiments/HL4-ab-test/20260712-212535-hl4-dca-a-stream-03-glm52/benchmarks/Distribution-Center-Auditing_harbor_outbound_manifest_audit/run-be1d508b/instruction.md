# Task Instruction

Execute the following steps in a single Python script to produce `/root/Outbound_Load_Audit.xlsx` and `/root/Outbound_Load_Brief.docx`.

## Step 0 — Inspect source files

Before writing any logic, read and print:
- All sheet names and the first 5 rows of `/root/Manifest_Plan.xlsx`
- All sheet names and the first 5 rows of `/root/Dock_Scan_Log.xlsx`
- All sheet names of `/root/Outbound_Audit_Template.xlsx`, and for each sheet print the first 5 rows

This lets you see the exact column names and data types. Print column names explicitly with `.columns.tolist()`.

## Step 1 — Load data

```python
import pandas as pd
from copy import copy
from openpyxl import load_workbook
from docx import Document

manifest = pd.read_excel('/root/Manifest_Plan.xlsx')
dock = pd.read_excel('/root/Dock_Scan_Log.xlsx')
```

Print `manifest.columns.tolist()`, `dock.columns.tolist()`, `manifest.shape`, `dock.shape`, and `manifest.dtypes`, `dock.dtypes`.

Map the manifest columns to the canonical 8 columns:
1. Shipment ID
2. Carton ID
3. Planned Zone
4. Route
5. Expected Weight
6. Hazmat Flag
7. Carrier
8. Wave

If the source column names differ (e.g., spaces, casing), rename them. Print the mapping you use.

Similarly, identify in the dock scan log the columns for: Shipment ID, Carton ID, Status, Scanned Zone (and any timestamp/sequence column). Print the mapping.

**Critical**: Convert `Shipment ID` and `Carton ID` to strings (`.astype(str).str.strip()`) in both dataframes to avoid type-mismatch during merge/lookup.

## Step 2 — Derive scan status

From `dock`, filter to rows where `Status == 'LOADED'` (check exact string; print unique Status values first).

For each `(Shipment ID, Carton ID)` group among LOADED rows, keep only the **latest** row. "Latest" means:
- If there is a timestamp column, sort by it descending and take the first.
- If there is an index/sequence column, use that.
- Otherwise, take the last row in file order (i.e., the one with the highest original index).

Call this filtered dataframe `loaded_scans`. Print its shape and first few rows.

## Step 3 — Build Formatted Data

Start with `manifest` (same row order). Add four columns:

```python
# Merge to find loaded scan for each manifest row
merged = manifest.merge(
    loaded_scans[['Shipment ID', 'Carton ID', 'Scanned Zone']],
    on=['Shipment ID', 'Carton ID'],
    how='left',
    indicator=True
)

merged['Missing Load Scan'] = (merged['_merge'] == 'left_only').astype(int)
merged['Zone Mismatch'] = (
    (merged['_merge'] == 'both') &
    (merged['Scanned Zone'].astype(str).str.strip() != merged['Planned Zone'].astype(str).str.strip())
).astype(int)
merged['Total Errors'] = merged['Missing Load Scan'] + merged['Zone Mismatch']
```

For `Error Summary`:
```python
def error_summary(row):
    parts = []
    if row['Missing Load Scan'] == 1:
        parts.append('Missing Load Scan')
    if row['Zone Mismatch'] == 1:
        parts.append('Zone Mismatch')
    return ', '.join(parts) if parts else 'None'

merged['Error Summary'] = merged.apply(error_summary, axis=1)
```

Drop the `_merge` and `Scanned Zone` columns. The final `Formatted Data` should have exactly 12 columns in the order specified. Print the first 10 rows and all unique `Error Summary` values.

## Step 4 — Build Summary

Group `merged` by `(Route, Shipment ID)` and sum `Missing Load Scan`, `Zone Mismatch`, `Total Errors`. Rename columns to: `Missing Load Scans`, `Zone Mismatches`, `Total Errors`.

Filter to groups where `Total Errors > 0`.

Sort by `Route` ascending, then `Shipment ID` ascending.

Append a Grand Total row:
- Route = 'Grand Total'
- Shipment ID = '-'
- remaining = column sums of the filtered groups

Print the full summary table.

## Step 5 — Write Excel

Load the template:
```python
wb = load_workbook('/root/Outbound_Audit_Template.xlsx')
```

Preserve the `Overview` sheet — do NOT modify it.

Create (or get) sheets `RawData`, `Formatted Data`, `Summary`. If they already exist in the template, clear their contents but keep the sheet. If not, create them.

Write data using openpyxl (not pandas ExcelWriter) to avoid accidentally modifying Overview:

### RawData sheet
Write headers from `manifest.columns` in row 1, then data rows starting row 2. Write concrete values (convert numpy types to Python native).

### Formatted Data sheet
Write the 12 column headers in row 1, then data rows. All values must be concrete (no formulas).

### Summary sheet
Write the 5 column headers in row 1, then data rows including the Grand Total row.

Save as `/root/Outbound_Load_Audit.xlsx`.

**Verify**: Re-read the saved file with openpyxl, print sheet names, and for each of `RawData`, `Formatted Data`, `Summary`, print the first 3 data rows and the total row count.

## Step 6 — Write Word Brief

Compute totals from the summary Grand Total row:
- total_missing = grand total Missing Load Scans
- total_zone = grand total Zone Mismatches  
- total_errors = grand total Total Errors

Identify the top 2 Shipment IDs with the most Total Errors (from the summary table, excluding Grand Total). Break ties by Shipment ID ascending.

Create `/root/Outbound_Load_Brief.docx` with a single executive summary paragraph (3-6 sentences) that includes:
1. Plain-language definition of Missing Load Scan: "A Missing Load Scan indicates a planned carton that was never scanned as loaded at the dock."
2. Plain-language definition of Zone Mismatch: "A Zone Mismatch indicates a carton scanned as loaded in a zone different from the planned zone."
3. The computed totals: mention the exact numbers for Missing Load Scans, Zone Mismatches, and Total Errors.
4. Mention at least two high-priority Shipment IDs (use the format `Shipment ID XXXX` or just the ID value) — use the top 2 identified above.
5. At least one actionable recommendation.

**Critical for test matching**: Make sure each Shipment ID value appears as a standalone token in the text (e.g., `SHP-1001` not embedded in another word). Also ensure the numeric totals appear as plain numbers.

Print the full text of the document after saving to verify.

## Step 7 — Final verification

Re-read both output files and print:
- Excel: sheet names, row counts per sheet, first 2 rows of each data sheet
- Word: full paragraph text

Confirm:
- Overview sheet is unchanged (compare with template)
- RawData has same number of rows as manifest
- Formatted Data has same number of rows as manifest and 12 columns
- Summary has only error groups + Grand Total
- Word doc mentions both check definitions, all three totals, and at least 2 Shipment IDs

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=hard, tags=[excel, openpyxl, docx, audit, logistics].
Verifier config: timeout_sec=900.0.