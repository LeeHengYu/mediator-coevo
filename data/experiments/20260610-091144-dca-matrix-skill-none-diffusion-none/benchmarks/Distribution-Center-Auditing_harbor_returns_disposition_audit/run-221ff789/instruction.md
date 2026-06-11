# Task Instruction

## Task: Distribution Center Returns Disposition Audit

You must produce two output files:
1. `/root/Returns_Disposition_Audit.xlsx`
2. `/root/Returns_Disposition_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect Input Files
- Read `/root/Return_Plan.xlsx` and print its sheet names, column headers, first 5 rows, and total row count.
- Read `/root/Disposition_Event_Log.xlsx` and print its sheet names, column headers, first 5 rows, and total row count.
- Read `/root/Disposition_Alias.xlsx` and print its sheet names, column headers, first 5 rows, and total row count.
- Carefully note the exact column names (including casing and spaces) in each file. Use those exact names when referencing columns in code.

#### Step 1: Build the audit logic in Python

Use `openpyxl` for Excel writing and `python-docx` for the Word document. Use `pandas` for data manipulation.

##### 1a: Load data
```python
import pandas as pd
plan = pd.read_excel('/root/Return_Plan.xlsx')
events = pd.read_excel('/root/Disposition_Event_Log.xlsx')
aliases = pd.read_excel('/root/Disposition_Alias.xlsx')
```
Print columns of each dataframe to confirm exact header names before proceeding.

##### 1b: Prepare alias mapping
- Build a dictionary from `Disposition_Alias.xlsx` that maps each alias (lowercased) to its standard disposition (lowercased). Print the alias dictionary to verify.

##### 1c: Filter events to COMPLETED only
- Filter `events` to rows where `Event Status` (use the exact column name found) equals `COMPLETED` (case-insensitive comparison).
- For each `(Return ID, Line ID)` group, keep only the row with the latest event. If there is a timestamp/date column, sort by it descending and take the first. If there is no explicit timestamp, keep the last row (highest index) per group as the latest.
- Print the number of kept COMPLETED events.

##### 1d: Normalize Final Disposition
- For each kept event row, take the `Final Disposition` value (use exact column name), lowercase it, look it up in the alias dictionary. If found, replace with the standard disposition. Otherwise keep the original.
- Store the normalized disposition.

##### 1e: Merge plan with events
- Left-merge the plan table with the kept events on `(Return ID, Line ID)`.
- Compute the four new columns:
  - `Missing Final Event`: 1 if no matched COMPLETED event, else 0
  - `Disposition Mismatch`: 1 if a matched event exists AND normalized Final Disposition (lowercased) != Planned Disposition (lowercased), else 0
  - `Total Errors`: sum of the two above
  - `Error Summary`: exactly one of `None`, `Missing Final Event`, `Disposition Mismatch`, or `Missing Final Event, Disposition Mismatch` based on which flags are 1

##### 1f: Validate
- Print value counts for `Missing Final Event`, `Disposition Mismatch`, `Total Errors`, and `Error Summary` to verify correctness.
- Print a few example rows where errors exist.

#### Step 2: Write `/root/Returns_Disposition_Audit.xlsx`

Use `openpyxl` to create a workbook with exactly three sheets named `RawData`, `Formatted Data`, `Summary`.

##### Sheet `RawData`
- Copy the plan table from `Return_Plan.xlsx` exactly (all original columns, same row order, same headers).

##### Sheet `Formatted Data`
- Same row order as RawData.
- Columns 1-8 must be exactly: `Return ID`, `Line ID`, `Planned Disposition`, `Reason Code`, `Requested Qty`, `Warehouse`, `Carrier`, `Lane`
- Columns 9-12 must be exactly: `Missing Final Event`, `Disposition Mismatch`, `Total Errors`, `Error Summary`
- Write concrete values (integers for numeric columns, strings for Error Summary). Do NOT write Excel formulas.

##### Sheet `Summary`
- Headers exactly: `Warehouse`, `Carrier`, `Missing Final Events`, `Disposition Mismatches`, `Total Errors`
- Group `Formatted Data` by `(Warehouse, Carrier)`, summing `Missing Final Event` → `Missing Final Events`, `Disposition Mismatch` → `Disposition Mismatches`, `Total Errors` → `Total Errors`.
- Include ONLY groups where `Total Errors > 0`.
- Sort by `Warehouse` ascending then `Carrier` ascending.
- Append a final Grand Total row: `Warehouse`=`Grand Total`, `Carrier`=`-`, and the remaining columns are dataset-wide totals.

After writing, re-read the file and print sheet names, headers, and row counts to verify.

#### Step 3: Write `/root/Returns_Disposition_Brief.docx`

Using `python-docx`, create a Word document with a short executive summary (3-6 sentences) that includes:
- A plain-language definition of both checks: "Missing Final Event" means no completed disposition event was recorded for a return line; "Disposition Mismatch" means the final recorded disposition differs from the planned disposition.
- The computed totals for Missing Final Events, Disposition Mismatches, and Total Errors (use the actual numbers from your computation).
- At least one actionable recommendation (e.g., implement real-time alerts for missing events, retrain staff on disposition codes).
- Mention at least two specific Return IDs that have the most errors (find the Return IDs with the highest Total Errors count across their lines and name them explicitly).

Save as `/root/Returns_Disposition_Brief.docx`.

#### Step 4: Final Verification
- Confirm both files exist: `/root/Returns_Disposition_Audit.xlsx` and `/root/Returns_Disposition_Brief.docx`
- Re-read the Excel file and print:
  - Sheet names (must be exactly `RawData`, `Formatted Data`, `Summary`)
  - `RawData` headers and row count
  - `Formatted Data` headers (must be exactly the 12 specified) and row count (must match RawData)
  - `Summary` headers (must be exactly the 5 specified), row count, and last row content (Grand Total row)
- Re-read the Word document and print its paragraph text to confirm it contains the required elements.

### Critical Reminders
- Column names in the output sheets must match EXACTLY as specified (case, spacing, order).
- Sheet names must match EXACTLY: `RawData`, `Formatted Data`, `Summary`.
- `Error Summary` values must be one of the four exact strings specified.
- The Summary sheet must only include (Warehouse, Carrier) groups with Total Errors > 0.
- The Grand Total row must use `Grand Total` and `-` as specified.
- All numeric audit columns must contain concrete integer values, not formulas.
- When normalizing dispositions, apply alias mapping case-insensitively, then compare case-insensitively.
- Install any needed packages (`pip install openpyxl python-docx pandas`) at the start if not already available.

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