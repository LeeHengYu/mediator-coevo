# Task Instruction

Build a cycle-count variance audit consisting of two deliverables: `/root/Cycle_Count_Variance_Audit.xlsx` and `/root/Cycle_Count_Variance_Brief.docx`.

## Inputs
- `/root/Cycle_Plan.xlsx` — plan table.
- `/root/Count_Event_Log.xlsx` — count event log.
- `/root/Cycle_Template.xlsx` — contains the `Overview` sheet to preserve verbatim.

## Step 1: Inspect inputs
Before writing, load each input with openpyxl and pandas to confirm column names, sheet names, and data shape. Identify the plan-table sheet in `Cycle_Plan.xlsx` (likely the first/only sheet) and the columns in `Count_Event_Log.xlsx` (expect at least Facility, Session ID, Bin ID, Event Type, Count Qty, and a timestamp/order column).

## Step 2: Derive final counts from the event log
- Drop rows where any of Facility, Session ID, Bin ID, or Count Qty is blank.
- Filter to rows where `Event Type` equals `FINAL` (case-sensitive match as specified; if data shows mixed case, confirm by inspection — default to exact `FINAL`).
- For each `(Facility, Session ID, Bin ID)`, keep only the latest row. Use a timestamp column if present; otherwise use the last occurrence in file order. Inspect column names to pick the correct ordering key.
- Result: a mapping from `(Facility, Session ID, Bin ID)` → `Count Qty`.

## Step 3: Build the workbook `/root/Cycle_Count_Variance_Audit.xlsx`
Create sheets in this exact order: `Overview`, `RawData`, `Formatted Data`, `Summary`.

### Overview
- Copy the `Overview` sheet from `Cycle_Template.xlsx` exactly, preserving values, merged cells, styles, column widths, row heights, and any images/shapes as best as openpyxl permits. Use openpyxl to copy the sheet from the loaded template workbook (e.g., load template, then copy worksheet cells and formatting) rather than re-typing content.

### RawData
- Copy the plan table from `Cycle_Plan.xlsx` exactly, including headers and row order.

### Formatted Data
- Same row order as `RawData`.
- Columns 1–7 exactly: Facility, Session ID, Bin ID, Product ID, Expected Qty, Allowed Variance, Approval Needed.
- Columns 8–11 with headers exactly: `Missing Final Count`, `Approval Gap`, `Total Errors`, `Error Summary`.
- For each row, compute:
  - `Missing Final Count` = 1 if no kept FINAL event exists for `(Facility, Session ID, Bin ID)`, else 0.
  - `Approval Gap` = 1 iff kept final exists AND `Approval Needed` is `YES` (case-insensitive) AND `abs(Expected Qty - Count Qty) > Allowed Variance`; else 0.
  - `Total Errors` = sum of the two flags.
  - `Error Summary` = exactly one of: `None`, `Missing Final Count`, `Approval Gap`, `Missing Final Count, Approval Gap` (use that exact comma+space ordering).
- Write concrete numeric/text values, not formulas.

### Summary
- Headers exactly: Facility, Session ID, Missing Final Counts, Approval Gaps, Total Errors.
- Aggregate from `Formatted Data` grouped by `(Facility, Session ID)`, summing the two flag columns and Total Errors.
- Include only groups where `Total Errors > 0`.
- Sort by Facility ascending, then Session ID ascending.
- Append a final row: Facility = `Grand Total`, Session ID = `-`, remaining columns = dataset totals across all included groups (sum of the summary rows you emitted, which equals dataset totals over rows with errors).

## Step 4: Build `/root/Cycle_Count_Variance_Brief.docx`
Write a 3–6 sentence executive summary using python-docx. It must contain:
- Plain-language definitions of both `Missing Final Count` and `Approval Gap` checks.
- The computed totals for Missing Final Counts, Approval Gaps, and Total Errors (use the Grand Total row values).
- At least one actionable recommendation.
- Mention at least two high-priority facility-session combinations with frequent exceptions. For each such mention, include BOTH formats in the text: the compact form `FACILITY-SESSION` and the verbose form `Facility FACILITY Session SESSION` (substituting actual values). Pick the top two `(Facility, Session ID)` groups by Total Errors from the Summary (break ties by Facility asc, then Session ID asc).

## Validation before finishing
- Open the produced xlsx and confirm sheet order is exactly [`Overview`, `RawData`, `Formatted Data`, `Summary`].
- Confirm `Formatted Data` headers match exactly and have 11 columns.
- Confirm `Summary` headers match exactly and the final row is `Grand Total`/`-` with correct totals.
- Confirm `Error Summary` values are only from the allowed four strings.
- Open the docx and verify it contains both the compact `FACILITY-SESSION` substring and the verbose `Facility FACILITY Session SESSION` substring for each of the two highlighted combos, plus the three numeric totals.
- Confirm `Overview` content is unchanged from the template (spot-check a few cells).

Do not modify the template file. Do not rename any sheets or output files.

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