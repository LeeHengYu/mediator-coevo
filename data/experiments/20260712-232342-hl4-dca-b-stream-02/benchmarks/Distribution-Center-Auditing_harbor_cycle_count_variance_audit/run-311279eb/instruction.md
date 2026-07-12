# Task Instruction

## Task: Build Cycle Count Variance Audit deliverables

You must produce two files:
1. `/root/Cycle_Count_Variance_Audit.xlsx`
2. `/root/Cycle_Count_Variance_Brief.docx`

### Step-by-step plan

#### Step 0 — Inspect source files
1. Open and inspect `/root/Cycle_Plan.xlsx` — list all sheet names, then print all rows of the plan table (note column names exactly).
2. Open and inspect `/root/Count_Event_Log.xlsx` — list all sheet names, print column names, print all rows. Pay close attention to the `Event Type` column values (exact casing/spacing) and note any rows with blank keys or blank `Count Qty`.
3. Open and inspect `/root/Cycle_Template.xlsx` — list all sheet names. For the `Overview` sheet, note its structure so you can copy it exactly.

#### Step 1 — Build the processing script

Write a Python script `/root/build_audit.py` that does everything below. Use `openpyxl` for Excel and `python-docx` for Word. Install them first if needed (`pip install openpyxl python-docx`).

##### 1a) Copy Overview sheet
- Copy the `Overview` sheet from `Cycle_Template.xlsx` cell-by-cell into the output workbook as the first sheet named `Overview`. Preserve merged cells, values, and formatting as closely as possible. Do NOT modify any content.

##### 1b) RawData sheet
- Copy the plan table from `Cycle_Plan.xlsx` exactly (all columns, all rows, same order) into a sheet named `RawData`.

##### 1c) Formatted Data sheet
- Start with the same rows as RawData, same order.
- The first 7 columns must be exactly: `Facility`, `Session ID`, `Bin ID`, `Product ID`, `Expected Qty`, `Allowed Variance`, `Approval Needed`.
- Process `Count_Event_Log.xlsx`:
  - Filter to rows where `Event Type` equals `FINAL` (case-insensitive, stripped).
  - Remove rows where any of `Facility`, `Session ID`, `Bin ID` is blank/None/NaN, or where `Count Qty` is blank/None/NaN.
  - For each unique `(Facility, Session ID, Bin ID)` key, keep only the LAST row (by original row order in the log, i.e., highest row index) — this is the "latest" final event.
  - Build a lookup dict: key = `(Facility, Session ID, Bin ID)` → `Count Qty` (as a number).
- For each row in the plan:
  - `Missing Final Count`: 1 if the key `(Facility, Session ID, Bin ID)` is NOT in the lookup dict, else 0.
  - `Approval Gap`: 1 if ALL THREE conditions hold:
    1. Key IS in the lookup dict (i.e., Missing Final Count == 0).
    2. `Approval Needed` equals `YES` (case-insensitive, stripped).
    3. `abs(Expected Qty - Count Qty)` is STRICTLY GREATER than `Allowed Variance`.
    Otherwise 0.
  - `Total Errors` = `Missing Final Count` + `Approval Gap`.
  - `Error Summary`: exactly one of `None`, `Missing Final Count`, `Approval Gap`, or `Missing Final Count, Approval Gap` (use this exact comma-space separator).
- Write concrete values (not formulas) for columns 8-11.

**CRITICAL**: When matching keys between the plan and the event log, ensure you compare values with matching types. Convert Facility, Session ID, and Bin ID to stripped strings for comparison. Also ensure numeric comparisons for Expected Qty, Count Qty, and Allowed Variance use numeric types.

##### 1d) Summary sheet
- Headers: `Facility`, `Session ID`, `Missing Final Counts`, `Approval Gaps`, `Total Errors`.
- Aggregate from Formatted Data by `(Facility, Session ID)`: sum `Missing Final Count` → `Missing Final Counts`, sum `Approval Gap` → `Approval Gaps`, sum `Total Errors` → `Total Errors`.
- Include ONLY groups where `Total Errors > 0`.
- Sort by `Facility` ascending then `Session ID` ascending.
- Append a Grand Total row: `Facility` = `Grand Total`, `Session ID` = `-`, remaining columns = sums of all included rows.

##### 1e) Word document
- Create `/root/Cycle_Count_Variance_Brief.docx` with a short executive summary (3-6 sentences) that:
  - Defines both checks in plain language: "Missing Final Count flags bins where no final count event was recorded" and "Approval Gap flags bins where the variance between expected and counted quantities exceeded the allowed threshold and required approval".
  - States the computed totals: X Missing Final Counts, Y Approval Gaps, Z Total Errors (use the Grand Total numbers).
  - Gives at least one actionable recommendation (e.g., "prioritize recounting bins with missing finals").
  - Mentions at least two specific high-priority `(Facility, Session ID)` combinations from the Summary sheet that have the most errors.

#### Step 2 — Run the script
Execute `python /root/build_audit.py` and check for errors.

#### Step 3 — Validate outputs
1. Re-open `/root/Cycle_Count_Variance_Audit.xlsx` and verify:
   - Sheet names are exactly `Overview`, `RawData`, `Formatted Data`, `Summary`.
   - `Formatted Data` has 11 columns with correct headers.
   - Spot-check a few rows: verify Missing Final Count and Approval Gap logic.
   - `Summary` has correct headers, only rows with Total Errors > 0, sorted correctly, Grand Total row present.
2. Re-open `/root/Cycle_Count_Variance_Brief.docx` and print all text to verify it contains the required elements.
3. If any issues are found, fix and re-run.

#### Key lessons from prior failures (avoid these mistakes)
- Do NOT get missing-event flags wrong. Double-check key matching (string normalization, type consistency).
- Do NOT get aggregate totals wrong in Summary. Verify sums match.
- Do NOT omit computed totals from the Word document. The exact numbers must appear in the text.
- Ensure the Grand Total row sums are correct dataset-wide totals (sum of all Summary rows, which equals sum of all Formatted Data rows).

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