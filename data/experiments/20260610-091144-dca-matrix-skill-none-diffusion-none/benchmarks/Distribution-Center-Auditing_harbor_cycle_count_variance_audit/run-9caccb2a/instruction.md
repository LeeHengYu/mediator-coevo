# Task Instruction

Create a Python script `/root/solve.py` and execute it to produce both deliverables. Follow these steps precisely:

## Step 1: Inspect source files
- Open and print the structure (sheet names, column headers, first few rows, row counts) of:
  - `/root/Cycle_Plan.xlsx`
  - `/root/Count_Event_Log.xlsx`
  - `/root/Cycle_Template.xlsx`
- Print all unique values in the `Event Type` column of `Count_Event_Log.xlsx` to understand exact casing/values.
- Print the `Overview` sheet content from `Cycle_Template.xlsx` to understand what must be preserved.

## Step 2: Write the solution script `/root/solve.py`

The script must:

### A) Read source data
- Read `Cycle_Plan.xlsx` into a DataFrame (the plan table).
- Read `Count_Event_Log.xlsx` into a DataFrame (the event log).
- Read `Cycle_Template.xlsx` with openpyxl to preserve the `Overview` sheet exactly.

### B) Build the `RawData` sheet
- This is an exact copy of the plan table from `Cycle_Plan.xlsx`.

### C) Build the `Formatted Data` sheet
- Start with the plan table (same row order as RawData).
- Ensure the first 7 columns are exactly: `Facility`, `Session ID`, `Bin ID`, `Product ID`, `Expected Qty`, `Allowed Variance`, `Approval Needed`.
- From `Count_Event_Log.xlsx`, filter to rows where `Event Type` equals `FINAL` (case-insensitive match). Drop rows where any of `Facility`, `Session ID`, `Bin ID`, or `Count Qty` is blank/NaN.
- For each unique `(Facility, Session ID, Bin ID)` group, keep only the LAST row (latest by row order in the event log, or by timestamp if a timestamp column exists). Extract the `Count Qty` from that row.
- Left-merge the plan table with these final counts on `(Facility, Session ID, Bin ID)`.
- Compute columns 8-11 as concrete values (NOT formulas):
  - `Missing Final Count`: 1 if no matching FINAL event was found, else 0.
  - `Approval Gap`: 1 if ALL three conditions hold: (a) a FINAL event exists (Missing Final Count == 0), (b) `Approval Needed` is `YES` (case-insensitive), (c) `abs(Expected Qty - Count Qty) > Allowed Variance`. Otherwise 0.
  - `Total Errors` = `Missing Final Count` + `Approval Gap`.
  - `Error Summary`: exactly one of `None`, `Missing Final Count`, `Approval Gap`, or `Missing Final Count, Approval Gap` based on which flags are 1.
- Write integer values for numeric columns (not floats like 1.0).

### D) Build the `Summary` sheet
- Aggregate from Formatted Data by `(Facility, Session ID)`.
- Columns: `Facility`, `Session ID`, `Missing Final Counts`, `Approval Gaps`, `Total Errors`.
- Include only groups where `Total Errors > 0`.
- Sort by `Facility` ascending, then `Session ID` ascending.
- Append a Grand Total row: `Facility`=`Grand Total`, `Session ID`=`-`, and sums for the three numeric columns.

### E) Write `/root/Cycle_Count_Variance_Audit.xlsx`
- Use openpyxl.
- First, copy the `Overview` sheet from `Cycle_Template.xlsx` exactly (cell by cell, preserving values, merged cells if any, and formatting as much as possible). Name it `Overview`.
- Then write `RawData`, `Formatted Data`, and `Summary` sheets.
- Ensure sheet order is: `Overview`, `RawData`, `Formatted Data`, `Summary`.

### F) Write `/root/Cycle_Count_Variance_Brief.docx`
- Use python-docx.
- Write 3-6 sentences that include:
  - Plain-language definitions of both checks: explain what `Missing Final Count` means (a planned bin had no final count recorded) and what `Approval Gap` means (the variance between expected and counted quantity exceeded the allowed threshold for a bin requiring approval).
  - The computed Grand Total numbers for Missing Final Counts, Approval Gaps, and Total Errors.
  - At least one actionable recommendation.
  - **CRITICAL**: Identify the top 2 (Facility, Session ID) combinations with the highest Total Errors from the Summary data. Mention them using HYPHENATED format like `FACILITY-SESSIONID` (e.g., `WHC-SC009`). Use exactly this format: join Facility and Session ID with a hyphen. This is required for the verifier.

## Step 3: Run the script
```bash
pip install openpyxl python-docx pandas
python /root/solve.py
```

## Step 4: Validate outputs
- Confirm `/root/Cycle_Count_Variance_Audit.xlsx` exists and has exactly 4 sheets: `Overview`, `RawData`, `Formatted Data`, `Summary`.
- Print the first few rows and shape of `Formatted Data` and `Summary` sheets.
- Print the Grand Total row from Summary.
- Print the full text content of the Word document to verify it contains hyphenated facility-session references and all required elements.
- If any issue is found, fix and re-run.

## Key Pitfalls to Avoid (from cross-task feedback)
- Do NOT use formats like `FAC (Session SESS)` or `FAC / SESS` in the Word doc. Use `FAC-SESS` hyphenated format.
- When filtering FINAL events, use case-insensitive comparison. Don't drop plan rows that have no matching event — flag them as Missing Final Count instead.
- The merge must be a LEFT join from plan to events, so every plan row appears in the output.
- Write concrete integer values, not formulas or floats.
- Preserve the Overview sheet from the template unchanged — copy cell by cell.

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