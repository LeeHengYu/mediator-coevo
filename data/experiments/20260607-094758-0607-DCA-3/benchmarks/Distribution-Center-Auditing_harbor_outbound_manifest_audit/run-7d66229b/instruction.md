# Task Instruction

Build an outbound carton handoff audit deliverable from three inputs at /root/: Manifest_Plan.xlsx, Dock_Scan_Log.xlsx, and Outbound_Audit_Template.xlsx. Produce two outputs: /root/Outbound_Load_Audit.xlsx and /root/Outbound_Load_Brief.docx.

Steps:

1. Inspect inputs first.
   - Open Manifest_Plan.xlsx with openpyxl and pandas; record sheet name, headers, row count, and exact column order.
   - Open Dock_Scan_Log.xlsx; record its columns (expect at least Shipment ID, Carton ID, Status, Scanned Zone, and some timestamp/order column).
   - Open Outbound_Audit_Template.xlsx with openpyxl (keep_vba=False, but preserve formatting). Note all existing sheets, especially the `Overview` sheet which must be preserved byte-for-byte in content and order.

2. Build the output workbook by loading the template with openpyxl.load_workbook so the `Overview` sheet is preserved untouched. Do NOT recreate it from scratch. Remove any other pre-existing placeholder sheets only if they conflict with the required new sheet names, and add the three required sheets: `RawData`, `Formatted Data`, `Summary`.

3. RawData sheet:
   - Copy the manifest plan table exactly (headers + all rows, same column order, same values). Write via openpyxl cell-by-cell to avoid type coercion issues.

4. Formatted Data sheet:
   - First 8 columns must be exactly (in this order): Shipment ID, Carton ID, Planned Zone, Route, Expected Weight, Hazmat Flag, Carrier, Wave. Preserve RawData row order.
   - Compute scan status: from Dock_Scan_Log, filter rows where Status == 'LOADED'. For each (Shipment ID, Carton ID), keep only the latest LOADED row. Determine "latest" by the timestamp column if present (parse as datetime); if no timestamp column, use the last occurrence in file order. Ignore non-LOADED rows entirely when selecting the kept scan.
   - Add columns 9-12 with exact headers: `Missing Load Scan`, `Zone Mismatch`, `Total Errors`, `Error Summary`.
     - Missing Load Scan = 1 if no kept LOADED scan exists for (Shipment ID, Carton ID), else 0.
     - Zone Mismatch = 1 if a kept LOADED scan exists AND its Scanned Zone != Planned Zone, else 0. (If Missing Load Scan == 1, Zone Mismatch must be 0.)
     - Total Errors = Missing Load Scan + Zone Mismatch (integer).
     - Error Summary must be exactly one of: `None`, `Missing Load Scan`, `Zone Mismatch`, `Missing Load Scan, Zone Mismatch`.
   - Write concrete numeric/text values, no formulas.

5. Summary sheet:
   - Headers exactly: Route, Shipment ID, Missing Load Scans, Zone Mismatches, Total Errors.
   - Aggregate Formatted Data by (Route, Shipment ID), summing Missing Load Scan, Zone Mismatch, Total Errors.
   - Keep only groups with Total Errors > 0.
   - Sort by Route ascending, then Shipment ID ascending.
   - Append a final row: Route='Grand Total', Shipment ID='-', remaining columns = dataset totals across the included groups' source rows (i.e., totals over all Formatted Data rows, which equal the sum of the displayed group rows since excluded groups contribute 0). Use the sum of all Formatted Data rows to be safe.

6. Save as /root/Outbound_Load_Audit.xlsx. Then reopen it and verify: Overview sheet still present and unchanged; sheet names include Overview, RawData, Formatted Data, Summary; RawData row count matches manifest; Formatted Data has 12 columns with the exact headers; Summary headers exact and last row is Grand Total.

7. Word brief /root/Outbound_Load_Brief.docx using python-docx:
   - 3-6 sentences executive summary.
   - Define Missing Load Scan (no LOADED dock scan recorded for the carton) and Zone Mismatch (carton's LOADED scan zone differs from its Planned Zone).
   - State totals: total Missing Load Scans, total Zone Mismatches, total Total Errors (sums from Formatted Data).
   - At least one actionable recommendation (e.g., re-verify dock scanner coverage on affected routes; retrain loaders on zone assignments).
   - Name at least two high-priority Shipment IDs with the most exceptions (pick top 2 by Total Errors aggregated per Shipment ID; ties broken by Shipment ID ascending).

8. Final validation: load both output files and confirm they open without error; assert exact sheet names, exact column headers, value types (integers for the three count columns), and that Error Summary values are from the allowed set.

Constraints: do not modify Overview; keep filenames and sheet names exactly as specified; write literal values (no formulas) in computed columns.

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