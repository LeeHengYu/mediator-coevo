# Task Instruction

## Task: Returns Disposition Audit

You must produce two files:
1. `/root/Returns_Disposition_Audit.xlsx` (with sheets `RawData`, `Formatted Data`, `Summary`)
2. `/root/Returns_Disposition_Brief.docx` (executive summary)

### Step-by-step Plan

#### Step 0: Inspect the input files
```bash
pip install openpyxl python-docx pandas
```
Then write a Python script to read and print:
- All columns and first few rows of `/root/Return_Plan.xlsx`
- All columns and first few rows of `/root/Disposition_Event_Log.xlsx`
- All columns and rows of `/root/Disposition_Alias.xlsx`

Print shapes and column names. Understand the data before proceeding.

#### Step 1: Build the audit in a single Python script

Create `/root/build_audit.py` that does everything below:

##### 1a) Read all three input files into pandas DataFrames
- `plan_df` from `Return_Plan.xlsx`
- `event_df` from `Disposition_Event_Log.xlsx`
- `alias_df` from `Disposition_Alias.xlsx`

##### 1b) Build `RawData` sheet
- Copy `plan_df` exactly (same columns, same row order) into sheet named `RawData`.

##### 1c) Build `Formatted Data` sheet
- Start with the first 8 columns of `plan_df` in exact order: `Return ID`, `Line ID`, `Planned Disposition`, `Reason Code`, `Requested Qty`, `Warehouse`, `Carrier`, `Lane`.
- **Important**: Verify the actual column names from the file. If they differ slightly (spaces, casing), map them to the required names.

**Derive event status:**
- From `event_df`, filter rows where `Event Status` == `COMPLETED` (case-insensitive comparison on the status column).
- For each `(Return ID, Line ID)` group, keep only the row with the latest event (use whatever timestamp/sequence column exists — inspect the data to find it; likely a date or sequence column).
- This gives you the "kept event" per line.

**Normalize dispositions using alias table:**
- Build a lookup dict from `Disposition_Alias.xlsx`: map each alias (lowered) to its standard disposition (lowered).
- For each kept event's `Final Disposition`, check if `final_disposition.strip().lower()` is a key in the alias dict. If yes, use the mapped standard value. Otherwise use the raw value. All comparisons are case-insensitive.

**Compute the 4 new columns for each row in plan_df:**
- `Missing Final Event`: 1 if no kept COMPLETED event exists for that `(Return ID, Line ID)`, else 0.
- `Disposition Mismatch`: 1 if a kept event exists AND the normalized Final Disposition (lowered) != Planned Disposition (lowered, stripped), else 0. If no event exists, this is 0.
- `Total Errors` = `Missing Final Event` + `Disposition Mismatch`.
- `Error Summary`: exactly one of:
  - `None` (the string, not Python None)
  - `Missing Final Event`
  - `Disposition Mismatch`
  - `Missing Final Event, Disposition Mismatch`

Write these as concrete values (int for numeric, str for Error Summary). Do NOT use Excel formulas.

##### 1d) Build `Summary` sheet
- Aggregate from the Formatted Data by `(Warehouse, Carrier)`.
- Sum `Missing Final Event` → `Missing Final Events`, `Disposition Mismatch` → `Disposition Mismatches`, `Total Errors` → `Total Errors`.
- Include only groups where `Total Errors > 0`.
- Sort by `Warehouse` ascending, then `Carrier` ascending.
- Append a Grand Total row: `Warehouse`=`Grand Total`, `Carrier`=`-`, and the remaining columns are dataset-wide sums.
- Headers must be exactly: `Warehouse`, `Carrier`, `Missing Final Events`, `Disposition Mismatches`, `Total Errors`.

##### 1e) Write the Excel file
- Use `openpyxl` via `pandas.ExcelWriter` with `engine='openpyxl'`.
- Write to `/root/Returns_Disposition_Audit.xlsx` with sheet names exactly `RawData`, `Formatted Data`, `Summary`.
- Use `index=False` for all sheets.

##### 1f) Build the Word document
- Use `python-docx` to create `/root/Returns_Disposition_Brief.docx`.
- Write an executive summary paragraph (3-6 sentences) that includes:
  1. A plain-language definition of both checks: "Missing Final Event flags return lines with no completed disposition event recorded" and "Disposition Mismatch flags lines where the final recorded disposition differs from the planned disposition".
  2. The computed totals: "The audit identified X Missing Final Events, Y Disposition Mismatches, and Z Total Errors across all return lines."
  3. At least one actionable recommendation, e.g., "We recommend prioritizing re-processing of lines with missing events and investigating root causes of disposition mismatches at the warehouse level."
  4. **Critically**: Mention at least two specific high-priority Return IDs. To find these, compute which Return IDs have the most Total Errors (sum across their lines), pick the top 2+ and mention them explicitly, e.g., "High-priority returns requiring immediate attention include Return ID RET-XXXX and Return ID RET-YYYY, which account for the highest number of exceptions."

**WARNING from prior failures**: The verifier checks that the Word document mentions at least two high-priority Return IDs. Make sure you extract the actual Return ID values from the data (the ones with the most errors) and write them as literal strings in the document. The verifier likely searches for these ID strings in the document text.

#### Step 2: Run the script
```bash
python /root/build_audit.py
```

#### Step 3: Validate outputs
After running, verify:
1. `/root/Returns_Disposition_Audit.xlsx` exists and has exactly 3 sheets with correct names.
2. `RawData` sheet has the same number of rows and columns as the original plan.
3. `Formatted Data` has 12 columns with the exact required headers.
4. The numeric columns contain concrete int values (0 or 1 for Missing/Mismatch, 0-2 for Total Errors).
5. `Summary` sheet has 5 columns with exact headers, only rows with Total Errors > 0, sorted correctly, and a Grand Total row at the end.
6. `/root/Returns_Disposition_Brief.docx` exists and contains the required content.

Write a quick validation snippet that reads back the outputs and prints column names, row counts, sheet names, summary contents, and the Word document text to confirm everything is correct.

#### Step 4: Check for the test file
Look for any test files in the task directory (e.g., `/root/tests/`, or files matching `test_*.py`). If found, run them with `pytest` to pre-validate before submission.

### Key Pitfalls to Avoid
- Do NOT use Python `None` for the Error Summary field — use the literal string `"None"`.
- Do NOT leave formulas in cells — write computed values.
- Do NOT forget the Grand Total row in Summary.
- Do NOT forget to mention specific Return IDs in the Word doc (this was a failure mode in a similar task).
- Column names must match exactly (case, spacing).
- Sheet names must match exactly (`RawData`, `Formatted Data`, `Summary`).

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=hard, tags=[excel, openpyxl, docx, audit, returns].
Verifier config: timeout_sec=900.0.