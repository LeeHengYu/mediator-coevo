# Task Instruction

Produce two deliverables for an outbound carton-handoff audit:

1. `/root/Outbound_Load_Audit.xlsx` (start from `/root/Outbound_Audit_Template.xlsx`)
2. `/root/Outbound_Load_Brief.docx`

## Step 1: Inspect inputs
- Open `/root/Manifest_Plan.xlsx`, `/root/Dock_Scan_Log.xlsx`, and `/root/Outbound_Audit_Template.xlsx` with pandas/openpyxl.
- Print column names, dtypes, row counts, and a few sample rows for each. Confirm template sheet names (must include `Overview`) and note any existing placeholder sheets (`RawData`, `Formatted Data`, `Summary`).
- Verify the Dock_Scan_Log has columns including `Shipment ID`, `Carton ID`, `Status`, `Scanned Zone`, and a timestamp column used to determine "latest".

## Step 2: Build the workbook
Copy the template file to `/root/Outbound_Load_Audit.xlsx` first (preserving `Overview` exactly — do not rewrite it; use openpyxl to load and only modify/add the three target sheets). If `RawData`, `Formatted Data`, or `Summary` already exist as placeholders, replace their contents; do not touch `Overview`.

### RawData sheet
- Copy the manifest table from `Manifest_Plan.xlsx` verbatim (same columns, same row order, same values).

### Formatted Data sheet
- Same row order as RawData.
- First 8 columns exactly: `Shipment ID`, `Carton ID`, `Planned Zone`, `Route`, `Expected Weight`, `Hazmat Flag`, `Carrier`, `Wave`.
- Add columns 9-12 with headers exactly: `Missing Load Scan`, `Zone Mismatch`, `Total Errors`, `Error Summary`.
- Derive scan status:
  - Filter Dock_Scan_Log to `Status == 'LOADED'`.
  - For each `(Shipment ID, Carton ID)`, keep only the latest row by timestamp.
  - Left-join this kept-scan set onto the manifest by `(Shipment ID, Carton ID)`.
- Compute per row:
  - `Missing Load Scan` = 1 if no kept LOADED scan exists, else 0.
  - `Zone Mismatch` = 1 if a kept LOADED scan exists AND `Scanned Zone` != `Planned Zone`, else 0. (Must be 0 when Missing Load Scan = 1.)
  - `Total Errors` = sum of the two.
  - `Error Summary` ∈ {`None`, `Missing Load Scan`, `Zone Mismatch`, `Missing Load Scan, Zone Mismatch`} matching the flags exactly.
- Write concrete numeric/text values (no formulas).

### Summary sheet
- Headers exactly: `Route`, `Shipment ID`, `Missing Load Scans`, `Zone Mismatches`, `Total Errors`.
- Aggregate from `Formatted Data` grouped by `(Route, Shipment ID)`, summing the two error flags and Total Errors.
- Include only groups where `Total Errors > 0`.
- Sort by `Route` asc, then `Shipment ID` asc.
- Append final Grand Total row: `Route` = `Grand Total`, `Shipment ID` = `-`, remaining three columns = dataset-wide totals (sum across all included groups, which equals the totals over Formatted Data).

## Step 3: Word brief `/root/Outbound_Load_Brief.docx`
Using python-docx, write 3–6 sentences that include all of:
- Plain-language definition of `Missing Load Scan` (carton has no LOADED dock scan recorded) and `Zone Mismatch` (carton was loaded but scanned in a zone different from its planned zone).
- The computed totals for Missing Load Scans, Zone Mismatches, and Total Errors (use the Grand Total numbers).
- At least one actionable recommendation (e.g., retrain dock staff on zone routing, audit scanner coverage for specific routes).
- At least two high-priority Shipment IDs with the most exceptions — pick the top Shipment IDs from the Summary sheet by Total Errors (ties broken by Shipment ID).

## Step 4: Validation before finishing
- Reopen `/root/Outbound_Load_Audit.xlsx` and confirm:
  - Sheets present: `Overview`, `RawData`, `Formatted Data`, `Summary` (Overview unchanged vs template — compare cell values).
  - `Formatted Data` has exactly 12 columns with the required headers in order.
  - Row count of `Formatted Data` equals manifest row count; row order matches RawData.
  - For a few sample rows, verify Missing Load Scan / Zone Mismatch / Total Errors / Error Summary are internally consistent.
  - `Summary` headers match exactly; only Total Errors > 0 groups present; sorted correctly; Grand Total row last; column sums match Formatted Data totals.
- Reopen the docx and confirm it contains both definitions, the three totals, a recommendation, and at least two specific Shipment IDs.

Report: files written, row counts per sheet, the three totals, and the high-priority Shipment IDs mentioned.

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