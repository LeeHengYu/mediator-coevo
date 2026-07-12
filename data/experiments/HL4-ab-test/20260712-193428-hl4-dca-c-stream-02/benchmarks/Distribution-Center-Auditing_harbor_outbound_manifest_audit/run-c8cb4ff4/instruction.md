# Task Instruction

## Task: Outbound Load Audit for Harbor Distribution Center

You must produce two files:
1. `/root/Outbound_Load_Audit.xlsx`
2. `/root/Outbound_Load_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect all input files

```bash
pip install openpyxl python-docx pandas
```

Then use Python to inspect:
- `/root/Manifest_Plan.xlsx` — read all sheets, print column headers and first 5 rows, count total rows.
- `/root/Dock_Scan_Log.xlsx` — read all sheets, print column headers and first 5 rows, count total rows. Pay special attention to the `Status` column values and the `Scanned Zone` column.
- `/root/Outbound_Audit_Template.xlsx` — list all sheet names, print contents of the `Overview` sheet (preserve it exactly), and inspect any other sheets.

Print all of this information before proceeding.

#### Step 1: Build the workbook

Use `openpyxl` to:
1. Open `/root/Outbound_Audit_Template.xlsx`.
2. **Preserve the `Overview` sheet exactly** — do not modify it in any way.
3. Create (or overwrite if they exist) sheets named exactly: `RawData`, `Formatted Data`, `Summary`.

#### Step 2: Populate `RawData`
- Read the manifest plan table from `Manifest_Plan.xlsx` using pandas.
- Write it into the `RawData` sheet with headers in row 1 and data starting row 2.
- Copy all columns and rows exactly as they appear.

#### Step 3: Populate `Formatted Data`

Using pandas:
1. Load the manifest plan into a DataFrame (`manifest_df`).
2. Load the dock scan log into a DataFrame (`scan_df`).
3. Filter `scan_df` to only rows where `Status == 'LOADED'`.
4. Among the filtered rows, for each `(Shipment ID, Carton ID)` group, keep only the **last row** (latest). Use the natural row order (last occurrence) unless there's an explicit timestamp column, in which case sort by it and take the last.
5. Left-merge `manifest_df` with the kept scans on `(Shipment ID, Carton ID)`.
6. Compute new columns:
   - `Missing Load Scan`: 1 if no matching LOADED scan was found, else 0. Use integer type.
   - `Zone Mismatch`: 1 if a LOADED scan exists AND `Scanned Zone != Planned Zone`, else 0. Use integer type.
   - `Total Errors`: `Missing Load Scan + Zone Mismatch`. Integer.
   - `Error Summary`: Exactly one of these strings:
     - `"None"` if Total Errors == 0
     - `"Missing Load Scan"` if Missing Load Scan == 1 and Zone Mismatch == 0
     - `"Zone Mismatch"` if Missing Load Scan == 0 and Zone Mismatch == 1
     - `"Missing Load Scan, Zone Mismatch"` if both == 1
7. The output columns must be exactly in this order:
   `Shipment ID, Carton ID, Planned Zone, Route, Expected Weight, Hazmat Flag, Carrier, Wave, Missing Load Scan, Zone Mismatch, Total Errors, Error Summary`
8. **Important**: Map the first 8 columns from the manifest. The column names in the manifest may differ slightly (e.g., `Zone` vs `Planned Zone`). Inspect the actual column names and rename as needed to match the required headers exactly.
9. Write concrete values (not formulas) to the `Formatted Data` sheet.
10. **Print the value counts**: Print `Missing Load Scan` sum, `Zone Mismatch` sum, `Total Errors` sum. Print the first 10 rows of the formatted data. These totals will be needed for the Word doc.

#### Step 4: Populate `Summary`

1. From the `Formatted Data` DataFrame, group by `(Route, Shipment ID)` and sum `Missing Load Scan`, `Zone Mismatch`, `Total Errors`.
2. Filter to only groups where `Total Errors > 0`.
3. Sort by `Route` ascending, then `Shipment ID` ascending.
4. Rename columns to exactly: `Route, Shipment ID, Missing Load Scans, Zone Mismatches, Total Errors`.
5. Append a final row: `Route="Grand Total"`, `Shipment ID="-"`, and the remaining columns are the dataset-wide totals (sum of all error columns from `Formatted Data`, not just from the filtered groups — actually, since we're summing from Formatted Data which includes all rows, the grand total should equal the sum across ALL rows in Formatted Data).
6. Write to the `Summary` sheet with headers in row 1.

#### Step 5: Save the workbook

Save as `/root/Outbound_Load_Audit.xlsx`. Then re-open and verify:
- Sheet names include `Overview`, `RawData`, `Formatted Data`, `Summary`.
- `Overview` content is unchanged.
- `Formatted Data` has 12 columns with correct headers.
- `Summary` has the correct headers and a Grand Total row.
- Print the Summary sheet contents to verify.

#### Step 6: Create the Word document

Using `python-docx`, create `/root/Outbound_Load_Brief.docx` with:
- A heading (e.g., "Outbound Load Audit – Executive Brief")
- A 3-6 sentence executive summary paragraph that includes:
  1. A plain-language definition of **Missing Load Scan** (a carton in the manifest that was never scanned as loaded at the dock).
  2. A plain-language definition of **Zone Mismatch** (a carton that was scanned as loaded but at a different zone than planned).
  3. The exact computed totals: state the number of Missing Load Scans, Zone Mismatches, and Total Errors using the actual numbers you computed.
  4. Mention at least **two specific Shipment IDs** that have the highest error counts (identify these from the Summary sheet — pick the top 2 by Total Errors).
  5. At least one actionable recommendation (e.g., implement real-time scan alerts, retrain dock staff on zone assignments, add pre-departure zone verification).

**Critical**: Make sure the numeric totals written in the Word doc exactly match the Grand Total row in the Summary sheet. The verifier will check for these exact numbers as strings in the document.

#### Step 7: Final Validation

Re-read both output files and print:
- All sheet names in the Excel file
- Row counts per sheet
- The `Formatted Data` column headers
- The full `Summary` sheet contents
- The full text content of the Word document

Confirm everything matches the specification before finishing.

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