# Task Instruction

## Task: Outbound Load Audit for Harbor Distribution Center

You must produce two files:
1. `/root/Outbound_Load_Audit.xlsx`
2. `/root/Outbound_Load_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect all input files
- Open and inspect `/root/Manifest_Plan.xlsx` — note all sheet names, column headers, row count, and data types.
- Open and inspect `/root/Dock_Scan_Log.xlsx` — note all sheet names, column headers, row count, data types. Pay special attention to the `Status` column values (check exact casing), `Scanned Zone` column name, and how `Shipment ID` / `Carton ID` are represented.
- Open and inspect `/root/Outbound_Audit_Template.xlsx` — note all sheet names (especially `Overview`), their contents, and any pre-existing formatting.

Print column names and first few rows of each file. Print unique values of the `Status` column in the dock scan log.

#### Step 1: Prepare the data in Python using openpyxl and pandas

Use pandas for data processing and openpyxl for writing the final Excel file (to preserve the template's `Overview` sheet).

```python
import pandas as pd
from openpyxl import load_workbook
from copy import copy

# Load data
manifest = pd.read_excel('/root/Manifest_Plan.xlsx')
dock_log = pd.read_excel('/root/Dock_Scan_Log.xlsx')
```

#### Step 2: Build `RawData` sheet
- This is an exact copy of the manifest plan table. Keep all columns and rows as-is.

#### Step 3: Build `Formatted Data` sheet

**3a: Filter dock scan log to only LOADED status rows**
- Filter `dock_log` to rows where `Status` == `'LOADED'` (check exact casing from Step 0; match it precisely).
- Among these LOADED rows, for each unique `(Shipment ID, Carton ID)` pair, keep only the **latest** row. Use whatever timestamp/sequence column exists to determine recency. If there's no timestamp column, keep the last occurrence (highest index).
- Call this filtered DataFrame `loaded_scans`.

**3b: Merge manifest with loaded scans**
- Start with the manifest DataFrame (same row order as RawData).
- The first 8 columns must be exactly: `Shipment ID`, `Carton ID`, `Planned Zone`, `Route`, `Expected Weight`, `Hazmat Flag`, `Carrier`, `Wave`. Rename columns from the manifest if needed to match these exact names.
- Left-merge with `loaded_scans` on `(Shipment ID, Carton ID)` to bring in `Scanned Zone`.

**3c: Compute the 4 new columns**
- `Missing Load Scan`: integer 1 if no matching LOADED scan exists (i.e., `Scanned Zone` is NaN after merge), else integer 0.
- `Zone Mismatch`: integer 1 if a LOADED scan exists AND `Scanned Zone` != `Planned Zone`, else integer 0. (If no scan exists, this is 0.)
- `Total Errors`: integer = `Missing Load Scan` + `Zone Mismatch`.
- `Error Summary`: string, exactly one of:
  - `'None'` (if Total Errors == 0)
  - `'Missing Load Scan'` (if only missing scan)
  - `'Zone Mismatch'` (if only zone mismatch)
  - `'Missing Load Scan, Zone Mismatch'` (if both — note: this can only happen if Missing=1 AND Zone Mismatch=1, but since Zone Mismatch requires a scan to exist, both being 1 simultaneously is logically impossible. Still, implement the logic faithfully.)

**CRITICAL**: Write these 4 columns as concrete values (int for numeric, str for Error Summary), NOT as Excel formulas.

**CRITICAL**: Ensure the `Formatted Data` sheet has exactly 12 columns with the exact headers specified. The first 8 come from the manifest, columns 9-12 are the new ones.

#### Step 4: Build `Summary` sheet

- Group the `Formatted Data` by `(Route, Shipment ID)`.
- Sum `Missing Load Scan`, `Zone Mismatch`, and `Total Errors` for each group.
- Filter to only groups where `Total Errors > 0`.
- Sort by `Route` ascending, then `Shipment ID` ascending.
- Append a Grand Total row: Route=`'Grand Total'`, Shipment ID=`'-'`, and remaining columns = sum of all filtered rows' values.
- Headers must be exactly: `Route`, `Shipment ID`, `Missing Load Scans`, `Zone Mismatches`, `Total Errors`.
  - Note the plural forms in the Summary headers (`Missing Load Scans`, `Zone Mismatches`) differ from the Formatted Data headers (`Missing Load Scan`, `Zone Mismatch`). Use the exact names specified for each sheet.

#### Step 5: Write the Excel file

- Load the template workbook: `wb = load_workbook('/root/Outbound_Audit_Template.xlsx')`
- **Do NOT modify the `Overview` sheet in any way.**
- Write `RawData`, `Formatted Data`, and `Summary` as new sheets. Use openpyxl to write cell-by-cell or use a helper, but ensure:
  - Sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
  - All numeric error columns are written as Python int values (not float, not string).
  - `Error Summary` is written as a Python string.
  - All other data preserves its original type from the manifest (strings stay strings, numbers stay numbers). If dates exist, write them as-is or as strings — check what the manifest contains.
- Save as `/root/Outbound_Load_Audit.xlsx`.

**After saving, re-read the file and verify:**
- The `Overview` sheet still exists and has the same content as the template.
- `RawData` has the correct number of rows (should match manifest row count).
- `Formatted Data` has 12 columns with correct headers.
- `Summary` has 5 columns with correct headers, only error rows + Grand Total.
- Print the Grand Total row values.

#### Step 6: Create the Word document

Use `python-docx`:

```python
from docx import Document
```

Create `/root/Outbound_Load_Brief.docx` with an executive summary paragraph (3-6 sentences) that includes ALL of the following:

1. **Plain-language definition of both checks:**
   - `Missing Load Scan`: A carton listed in the manifest plan was not recorded as loaded in the dock scan log (no LOADED scan entry found).
   - `Zone Mismatch`: A carton was scanned as loaded but in a different zone than originally planned.

2. **Computed totals** — use the actual numbers from the Grand Total row:
   - Total Missing Load Scans: [number]
   - Total Zone Mismatches: [number]
   - Total Errors: [number]
   - Write these as digits in the text (e.g., "3 Missing Load Scans").

3. **At least one actionable recommendation** — e.g., "We recommend implementing real-time zone validation at scan points to prevent zone mismatches."

4. **At least two high-priority Shipment IDs** — identify the Shipment IDs with the highest Total Errors from the Summary sheet. Mention them explicitly by their exact Shipment ID values (e.g., "Shipment SH-1001 and Shipment SH-1005 had the most exceptions"). Use the exact IDs as they appear in the data. Mention at least 2.

**CRITICAL for the Word doc**: The verifier will search the document text for the shipment IDs. Make sure you use the EXACT shipment ID strings as they appear in the data (not paraphrased or reformatted). Include the full ID string.

#### Step 7: Final Verification

- Confirm `/root/Outbound_Load_Audit.xlsx` exists and has sheets: `Overview`, `RawData`, `Formatted Data`, `Summary`.
- Confirm `/root/Outbound_Load_Brief.docx` exists and contains the required content.
- Print the first 5 rows and last 2 rows of `Formatted Data`.
- Print all rows of `Summary`.
- Print the full text content of the Word document.

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