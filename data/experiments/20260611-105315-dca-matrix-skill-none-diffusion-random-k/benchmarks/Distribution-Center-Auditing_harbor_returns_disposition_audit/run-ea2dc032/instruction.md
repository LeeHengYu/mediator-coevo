# Task Instruction

## Task: Distribution Center Returns Disposition Audit

You must produce two output files:
1. `/root/Returns_Disposition_Audit.xlsx` (3 worksheets)
2. `/root/Returns_Disposition_Brief.docx` (executive summary)

### Step-by-step Instructions

#### Step 1: Inspect the input files

Read and display the contents of all three input files to understand their structure:
```python
import pandas as pd

plan = pd.read_excel('/root/Return_Plan.xlsx')
print('Return_Plan columns:', list(plan.columns))
print('Return_Plan shape:', plan.shape)
print(plan.head(10))
print(plan.dtypes)

events = pd.read_excel('/root/Disposition_Event_Log.xlsx')
print('\nEvent_Log columns:', list(events.columns))
print('Event_Log shape:', events.shape)
print(events.head(10))
print(events.dtypes)

alias = pd.read_excel('/root/Disposition_Alias.xlsx')
print('\nAlias columns:', list(alias.columns))
print('Alias shape:', alias.shape)
print(alias)
print(alias.dtypes)
```

Carefully note the exact column names in each file. Identify which columns correspond to: Return ID, Line ID, Planned Disposition, Reason Code, Requested Qty, Warehouse, Carrier, Lane, Event Status, Final Disposition, and any timestamp/sequence column for determining "latest" event.

#### Step 2: Build the audit logic in Python

Write a single Python script that does everything. Use `openpyxl` for Excel writing and `python-docx` for the Word document. Install if needed: `pip install openpyxl python-docx`.

The script must:

**A) RawData worksheet:**
- Copy the plan table from `Return_Plan.xlsx` exactly as-is (same columns, same rows, same order).

**B) Formatted Data worksheet:**

1. Start with the plan data. Keep same row order.
2. Ensure the first 8 columns are exactly named: `Return ID`, `Line ID`, `Planned Disposition`, `Reason Code`, `Requested Qty`, `Warehouse`, `Carrier`, `Lane`. If the source columns have different names, rename them to match (map carefully based on your inspection in Step 1).
3. Process the event log:
   - Filter to only rows where `Event Status` equals `COMPLETED` (case-insensitive comparison to be safe).
   - For each `(Return ID, Line ID)` group, keep only the latest row. Use whatever timestamp or sequence column exists to determine "latest". If there's a date/time column, sort by it descending and take the first. If there's an event sequence number, use the highest.
   - Extract the `Final Disposition` from the kept row.
4. Build the alias mapping:
   - Read `Disposition_Alias.xlsx`. It likely has columns like `Alias` and `Standard Disposition` (or similar). Create a dictionary mapping each alias (lowercased) to its standard disposition (lowercased).
5. For each plan row, compute:
   - Look up the kept COMPLETED event for that `(Return ID, Line ID)`.
   - `Missing Final Event` = 1 if no such event exists, else 0.
   - If event exists, normalize the `Final Disposition`: look up the lowercased value in the alias dict; if found, use the mapped standard; otherwise use the raw value. Compare (case-insensitive) to `Planned Disposition`.
   - `Disposition Mismatch` = 1 if event exists AND normalized final disposition != planned disposition (case-insensitive), else 0.
   - `Total Errors` = `Missing Final Event` + `Disposition Mismatch`.
   - `Error Summary`: exactly one of `None`, `Missing Final Event`, `Disposition Mismatch`, or `Missing Final Event, Disposition Mismatch` based on which flags are 1.
6. Write all 12 columns to the `Formatted Data` sheet. The four new columns (9-12) must contain concrete values (integers for the numeric ones, strings for Error Summary), NOT formulas.

**C) Summary worksheet:**

1. From the Formatted Data, group by `(Warehouse, Carrier)`.
2. For each group, sum `Missing Final Event`, `Disposition Mismatch`, and `Total Errors`.
3. Keep only groups where `Total Errors > 0`.
4. Sort by `Warehouse` ascending, then `Carrier` ascending.
5. Headers must be exactly: `Warehouse`, `Carrier`, `Missing Final Events`, `Disposition Mismatches`, `Total Errors`.
6. Append a Grand Total row: Warehouse=`Grand Total`, Carrier=`-`, and the remaining columns = sums across all data (not just filtered groups — use the full dataset totals).

**D) Word Document `/root/Returns_Disposition_Brief.docx`:**

1. Write an executive summary of 3-6 sentences.
2. Must include:
   - Plain-language definition of both checks: "Missing Final Event" means no completed disposition event was recorded for a return line; "Disposition Mismatch" means the final recorded disposition differs from the planned disposition.
   - The computed totals for Missing Final Events, Disposition Mismatches, and Total Errors (use the Grand Total numbers).
   - At least one actionable recommendation (e.g., "We recommend investigating warehouse X" or "Implement real-time disposition tracking").
   - Mention at least two specific Return IDs that have the most errors/exceptions. To find these, group the Formatted Data by Return ID, sum Total Errors, and pick the top 2.

#### Step 3: Run the script and verify

After running the script:
1. Re-read `/root/Returns_Disposition_Audit.xlsx` and verify:
   - Sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
   - `RawData` row count matches the original plan.
   - `Formatted Data` has 12 columns with correct headers.
   - `Formatted Data` numeric columns contain integers (0 or 1 for flags).
   - `Error Summary` values are only from the four allowed strings.
   - `Summary` has exactly 5 columns with correct headers.
   - `Summary` last row has Warehouse=`Grand Total` and Carrier=`-`.
   - Grand Total row sums match the actual totals from Formatted Data.
2. Re-read `/root/Returns_Disposition_Brief.docx` and print its text to verify it contains the required elements.

#### Step 4: Fix any issues

If verification reveals any problems (wrong column names, wrong counts, missing sheets, etc.), fix and re-run.

### Critical Details to Watch
- Column name matching: The source files may use slightly different column names. Map them carefully.
- The alias mapping must be applied case-insensitively. Convert both sides to lowercase for lookup.
- "Latest" COMPLETED event: identify the correct column to sort by (timestamp, event date, sequence number, etc.).
- The Summary sheet header says `Missing Final Events` and `Disposition Mismatches` (plural), while the Formatted Data columns say `Missing Final Event` and `Disposition Mismatch` (singular). Use the exact names specified for each sheet.
- Grand Total row: compute totals from ALL plan rows (the full Formatted Data), not just the filtered summary groups.
- Write concrete values, not Excel formulas, for the four derived columns.
- Output filenames must be exactly `/root/Returns_Disposition_Audit.xlsx` and `/root/Returns_Disposition_Brief.docx`.

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