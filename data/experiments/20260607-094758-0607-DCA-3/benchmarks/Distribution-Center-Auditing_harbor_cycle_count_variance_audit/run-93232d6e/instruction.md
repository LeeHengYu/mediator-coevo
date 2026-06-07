# Task Instruction

Build two deliverables for a cycle-count variance audit.

## Input Files
- `/root/Cycle_Plan.xlsx` — plan table
- `/root/Count_Event_Log.xlsx` — count events
- `/root/Cycle_Template.xlsx` — template with `Overview` sheet

## Deliverable 1: `/root/Cycle_Count_Variance_Audit.xlsx`

Must contain exactly these worksheets in this order: `Overview`, `RawData`, `Formatted Data`, `Summary`.

### Overview
- Copy the `Overview` sheet from `Cycle_Template.xlsx` exactly, preserving formatting, merged cells, styles, and values. Use openpyxl copy_worksheet or manually copy cell values + styles. Verify by inspecting the source sheet first.

### RawData
- Copy the plan table from `Cycle_Plan.xlsx` exactly (headers + all data rows, same order).

### Formatted Data
- Same row order as RawData.
- First 7 columns must be exactly (in this order):
  1. Facility
  2. Session ID
  3. Bin ID
  4. Product ID
  5. Expected Qty
  6. Allowed Variance
  7. Approval Needed
- Add columns 8–11 with these exact headers:
  8. Missing Final Count
  9. Approval Gap
  10. Total Errors
  11. Error Summary

Deriving the final count from `Count_Event_Log.xlsx`:
- Filter to rows where `Event Type == 'FINAL'`.
- Drop rows where any of `Facility`, `Session ID`, `Bin ID`, or `Count Qty` is blank/NaN.
- For each `(Facility, Session ID, Bin ID)` key, keep only the LATEST row. If there is a timestamp column (e.g., `Event Time`/`Timestamp`), use it; otherwise use the last occurrence in file order.

Rules per row in `Formatted Data`:
- `Missing Final Count` = 1 if no kept FINAL event exists for that `(Facility, Session ID, Bin ID)`; else 0.
- `Approval Gap` = 1 iff ALL: kept final exists AND `Approval Needed` equals 'YES' case-insensitively AND `abs(Expected Qty - Count Qty) > Allowed Variance`. Else 0.
- `Total Errors` = Missing Final Count + Approval Gap.
- `Error Summary` is exactly one of: `None`, `Missing Final Count`, `Approval Gap`, `Missing Final Count, Approval Gap`.
- Write CONCRETE numeric/text values (no formulas).

### Summary
Headers (exact order):
1. Facility
2. Session ID
3. Missing Final Counts
4. Approval Gaps
5. Total Errors

- Aggregate from `Formatted Data` grouped by `(Facility, Session ID)`.
- Include only groups where `Total Errors > 0`.
- Sort by Facility asc, then Session ID asc.
- Append a final row: `Facility='Grand Total'`, `Session ID='-'`, columns 3–5 = dataset totals.

## Deliverable 2: `/root/Cycle_Count_Variance_Brief.docx`

A 3–6 sentence executive summary that MUST include all of the following:
1. Plain-language definitions of both checks:
   - `Missing Final Count`: no FINAL count event was recorded for a planned bin.
   - `Approval Gap`: a final count exists, approval was required, and the absolute variance from expected exceeded the allowed tolerance.
2. The computed totals (integers) for Missing Final Counts, Approval Gaps, and Total Errors taken from the Summary Grand Total row.
3. At least one actionable recommendation (e.g., re-count missing bins, route flagged variances to supervisor approval).
4. Mention at least TWO high-priority Facility/Session combinations with the most exceptions.

### MANDATORY procedure for the high-priority mentions (this is where prior runs failed):
- Take the Summary rows (excluding the Grand Total row).
- Sort them by `Total Errors` DESCENDING (tiebreak: Missing Final Counts desc, then Facility asc, Session ID asc).
- Take the top 2 rows.
- For each, write the pair into the docx text in BOTH of these literal formats joined together to be safe: `FACILITY-SESSION` (hyphenated, e.g., `DC01-S123`) AND also as `Facility FACILITY Session SESSION`. For example, write a sentence like: "High-priority focus areas include DC01-S123 (Facility DC01 Session S123) and DC02-S045 (Facility DC02 Session S045), which account for the highest exception counts."
- Use the EXACT Facility and Session ID values as strings from the Summary sheet (do not reformat, pad, or alter them).
- If fewer than 2 groups exist with Total Errors > 0, still mention the top available, but log this; the task expects at least 2.

## Execution Steps
1. Inspect each input file's structure (sheet names, columns, sample rows) before writing code.
2. Inspect the template's `Overview` sheet to understand what must be preserved.
3. Build the workbook with openpyxl so the `Overview` sheet styles can be preserved; use pandas only for data manipulation, then write values via openpyxl.
4. Compute Summary aggregates from the Formatted Data values you just wrote (not from a separate recomputation path) to ensure consistency.
5. Derive the top-2 high-priority pairs from the Summary aggregates BEFORE generating the docx, and store them as plain strings.
6. Generate the docx with python-docx, ensuring the top-2 pairs appear in the document text in the `FACILITY-SESSION` format.
7. Validation before finishing:
   - Open the produced xlsx and confirm sheet names are exactly `Overview`, `RawData`, `Formatted Data`, `Summary` in that order.
   - Confirm `Formatted Data` has 11 columns with exact headers.
   - Confirm `Summary` ends with a `Grand Total` / `-` row and totals match column sums.
   - Open the produced docx and grep the text for both top-2 `FACILITY-SESSION` strings; assert each substring is present.
   - Confirm the docx contains the three integer totals.

## Constraints
- Do not alter the Overview sheet content.
- Filenames, sheet names, and column headers must match exactly.
- Write concrete values in added columns; no formulas.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=expert, tags=[excel, openpyxl, docx, audit, inventory].
Verifier config: timeout_sec=900.0.