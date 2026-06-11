# Task Instruction

Write and execute a Python script that performs the following steps:

1. **Read the source files:**
   - Read `/root/Manifest_Plan.xlsx` into a DataFrame (manifest).
   - Read `/root/Dock_Scan_Log.xlsx` into a DataFrame (scans).
   - Inspect `/root/Outbound_Audit_Template.xlsx` to understand its structure (especially the `Overview` sheet).

2. **Prepare the template copy:**
   - Use `shutil.copy` to copy `/root/Outbound_Audit_Template.xlsx` to `/root/Outbound_Load_Audit.xlsx`.
   - This preserves the `Overview` sheet exactly.

3. **Build the `RawData` sheet:**
   - Copy the manifest plan table exactly as-is from `Manifest_Plan.xlsx`.

4. **Build the `Formatted Data` sheet:**
   - Start with the same rows and order as `RawData`.
   - Normalize column names for matching (use case-insensitive mapping to handle any casing variations between the manifest and scan log).
   - Ensure the first 8 columns are exactly: `Shipment ID`, `Carton ID`, `Planned Zone`, `Route`, `Expected Weight`, `Hazmat Flag`, `Carrier`, `Wave`.
   - From the scan log, filter to rows where `Status == 'LOADED'`. Among those, for each `(Shipment ID, Carton ID)` group, keep only the latest row (by whatever timestamp/row-order column is available; if there's a timestamp column use it, otherwise use the last occurrence).
   - Add four new columns (9-12):
     - `Missing Load Scan`: 1 if no kept LOADED scan exists for that (Shipment ID, Carton ID), else 0.
     - `Zone Mismatch`: 1 if a kept LOADED scan exists AND `Scanned Zone != Planned Zone`, else 0.
     - `Total Errors`: `Missing Load Scan + Zone Mismatch`.
     - `Error Summary`: exactly one of `None`, `Missing Load Scan`, `Zone Mismatch`, or `Missing Load Scan, Zone Mismatch` based on which flags are set.
   - Write concrete values (not formulas) for columns 9-12.

5. **Build the `Summary` sheet:**
   - Aggregate from `Formatted Data` by `(Route, Shipment ID)`.
   - Columns: `Route`, `Shipment ID`, `Missing Load Scans`, `Zone Mismatches`, `Total Errors`.
   - Include only groups where `Total Errors > 0`.
   - Sort by `Route` ascending, then `Shipment ID` ascending.
   - Append a final row: `Route`=`Grand Total`, `Shipment ID`=`-`, and the remaining columns are dataset-wide totals.

6. **Write the Excel file:**
   - Open `/root/Outbound_Load_Audit.xlsx` with `pd.ExcelWriter` using `openpyxl` engine in **append** mode (`if_sheet_exists='replace'`).
   - Write `RawData`, `Formatted Data`, and `Summary` sheets. Do NOT touch the `Overview` sheet.
   - Use `index=False` for all sheets.

7. **Create the Word brief `/root/Outbound_Load_Brief.docx`:**
   - Use `python-docx` to create a document.
   - Add a heading "Outbound Load Audit Brief".
   - Write 3-6 sentences that include:
     - A plain-language definition of both checks: Missing Load Scan means a carton in the manifest was never scanned as loaded at the dock; Zone Mismatch means a carton was loaded but scanned in a different zone than planned.
     - The computed totals for Missing Load Scans, Zone Mismatches, and Total Errors (use the Grand Total values).
     - At least one actionable recommendation (e.g., implement real-time scan alerts, retrain dock workers on zone assignments).
     - Mention at least two specific high-priority Shipment IDs that have the most total errors.
   - Save to `/root/Outbound_Load_Brief.docx`.

8. **Validation:**
   - After writing, re-read `/root/Outbound_Load_Audit.xlsx` and print:
     - Sheet names (confirm `Overview`, `RawData`, `Formatted Data`, `Summary` all exist).
     - Shape and first few rows of each data sheet.
     - The Grand Total row from Summary.
   - Confirm `/root/Outbound_Load_Brief.docx` exists and print its paragraph texts.

**Important implementation notes:**
- Use case-insensitive column name matching when joining manifest and scan data.
- For the scan log, if there is a timestamp column, sort by it to pick the latest LOADED scan; otherwise use `.drop_duplicates(subset=[shipment_col, carton_col], keep='last')` after filtering to LOADED status.
- Ensure `Error Summary` uses the exact strings specified (with `None` as a string, not Python None/NaN).
- The `Overview` sheet must remain byte-identical to the template's version.

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