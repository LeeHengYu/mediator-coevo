# Task Instruction

Build two deliverables for a cycle-count variance audit.

INPUTS:
- /root/Cycle_Plan.xlsx (plan table)
- /root/Count_Event_Log.xlsx (count events)
- /root/Cycle_Template.xlsx (contains an `Overview` sheet to copy verbatim)

DELIVERABLE 1: /root/Cycle_Count_Variance_Audit.xlsx with EXACTLY these sheets in this order: `Overview`, `RawData`, `Formatted Data`, `Summary`.

Step 1 — Inspect inputs first.
- Open each input with openpyxl and pandas to confirm column names, sheet names, row counts, and data types BEFORE writing output.
- Identify the `Overview` sheet structure in Cycle_Template.xlsx (cells, merged ranges, styles, images if any). Use openpyxl to copy it cell-by-cell including values, styles, merged cells, column widths, and row heights so it is preserved unchanged.

Step 2 — RawData sheet.
- Copy the plan table from Cycle_Plan.xlsx exactly (same column order, same values, same row order). Header row + data rows only.

Step 3 — Build the final-count lookup from Count_Event_Log.xlsx.
- Filter to rows where `Event Type` == 'FINAL' (case-sensitive as given; if data shows mixed case, match exactly as specified — verify by inspection).
- Drop rows where any of `Facility`, `Session ID`, `Bin ID`, or `Count Qty` is blank/NaN.
- For each (Facility, Session ID, Bin ID), keep the LATEST row. Determine latest by the timestamp column present in the log (inspect column names; typical names: `Event Time`, `Timestamp`, `Event Timestamp`). If multiple timestamp-like columns exist, pick the one that orders FINAL events. If no timestamp exists, use last occurrence by row order.
- Result: dict keyed by (Facility, Session ID, Bin ID) -> Count Qty.

Step 4 — Formatted Data sheet.
- Preserve RawData row order.
- First 7 columns identical to RawData: Facility, Session ID, Bin ID, Product ID, Expected Qty, Allowed Variance, Approval Needed.
- Add columns 8–11 with headers exactly: `Missing Final Count`, `Approval Gap`, `Total Errors`, `Error Summary`.
- For each row compute concrete numeric/text values (NOT formulas):
  * Missing Final Count = 1 if (Facility, Session ID, Bin ID) not in final lookup, else 0.
  * Approval Gap = 1 iff final exists AND str(Approval Needed).strip().upper() == 'YES' AND abs(Expected Qty - Count Qty) > Allowed Variance; else 0.
  * Total Errors = Missing Final Count + Approval Gap.
  * Error Summary: one of 'None', 'Missing Final Count', 'Approval Gap', 'Missing Final Count, Approval Gap' based on which flags are 1.

Step 5 — Summary sheet.
- Headers exactly: `Facility`, `Session ID`, `Missing Final Counts`, `Approval Gaps`, `Total Errors`.
- Aggregate from Formatted Data by (Facility, Session ID): sums of Missing Final Count, Approval Gap, Total Errors.
- Include only groups with Total Errors > 0.
- Sort by Facility asc, then Session ID asc.
- Append final row: Facility='Grand Total', Session ID='-', then totals of the three numeric columns across the included rows (i.e., totals of the filtered groups, which equal dataset totals for those errors).

Step 6 — Validation before finishing.
- Reopen the produced workbook and assert: sheet names and order are exactly [`Overview`, `RawData`, `Formatted Data`, `Summary`]; RawData row count == Cycle_Plan row count; Formatted Data has 11 columns with the exact headers; every Error Summary value is in the allowed set; Total Errors == Missing Final Count + Approval Gap on every row; Summary grand total row equals sum of the rows above it.
- Verify the Overview sheet content matches the template (compare cell values cell-by-cell for the used range).

DELIVERABLE 2: /root/Cycle_Count_Variance_Brief.docx
- Use python-docx.
- 3–6 sentences total in an executive summary.
- Must include: plain-language definitions of `Missing Final Count` (no FINAL count event was recorded for a planned bin) and `Approval Gap` (a FINAL count exists but the variance from expected exceeds the allowed tolerance on a bin flagged as needing approval); the computed totals for Missing Final Counts, Approval Gaps, and Total Errors (use the grand totals from Summary); at least one concrete actionable recommendation; and explicit mention of at least two high-priority Facility / Session ID combinations with the most exceptions (pick top 2 by Total Errors from Summary, ties broken by Facility then Session ID).

CONSTRAINTS:
- Do not rename files or sheets.
- Do not use Excel formulas in the added columns; write resolved values.
- Preserve Overview sheet exactly (styles, merges, widths).
- Do not modify inputs.
- If a column name in inputs differs slightly from what is described, follow the actual input column names and map them to the required output headers exactly as specified.

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