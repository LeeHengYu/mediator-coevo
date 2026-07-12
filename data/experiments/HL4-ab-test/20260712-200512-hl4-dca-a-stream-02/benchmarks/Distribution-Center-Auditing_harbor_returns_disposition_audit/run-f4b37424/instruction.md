# Task Instruction

Execute the following steps in order.

## Step 1 – Inspect source files

Open and print the first few rows, all column names, and sheet names for each of:
- `/root/Return_Plan.xlsx`
- `/root/Disposition_Event_Log.xlsx`
- `/root/Disposition_Alias.xlsx`

Also print the number of rows in each file. This is critical for understanding the exact column names and data before writing the processing script.

## Step 2 – Write and run the Python processing script

Create and execute a Python script (`/root/solve.py`) that does the following:

### 2a – Load data
```python
import pandas as pd
from openpyxl import Workbook
from docx import Document

plan = pd.read_excel('/root/Return_Plan.xlsx')
events = pd.read_excel('/root/Disposition_Event_Log.xlsx')
alias = pd.read_excel('/root/Disposition_Alias.xlsx')
```
Print columns of each DataFrame to confirm names.

### 2b – Build alias mapping
From `Disposition_Alias.xlsx`, build a dict mapping each alias (lowercased) to its standard disposition (lowercased). Print the mapping for verification.

### 2c – Filter events to COMPLETED only, keep latest per (Return ID, Line ID)
- Filter `events` to rows where `Event Status` equals `COMPLETED` (case-insensitive strip comparison).
- Sort by whatever timestamp/sequence column exists (inspect the columns first) descending, then drop duplicates on `(Return ID, Line ID)` keeping first.
- This gives one row per (Return ID, Line ID) with the latest COMPLETED event.

### 2d – Build Formatted Data
- Start with the plan DataFrame (same row order).
- Keep exactly the first 8 columns as specified: `Return ID`, `Line ID`, `Planned Disposition`, `Reason Code`, `Requested Qty`, `Warehouse`, `Carrier`, `Lane`. Map from actual column names if they differ slightly.
- Left-merge with the filtered events on `(Return ID, Line ID)` to get `Final Disposition`.
- Normalize `Final Disposition`: lowercase it, look up in alias dict; if found, use the standard disposition (lowercased); otherwise keep the lowercased raw value.
- Also lowercase `Planned Disposition` for comparison.
- Compute:
  - `Missing Final Event` = 1 if no COMPLETED event (Final Disposition is NaN after merge), else 0
  - `Disposition Mismatch` = 1 if COMPLETED event exists AND normalized final disposition != planned disposition (both lowercased), else 0
  - `Total Errors` = Missing Final Event + Disposition Mismatch
  - `Error Summary` = exactly one of: `None`, `Missing Final Event`, `Disposition Mismatch`, `Missing Final Event, Disposition Mismatch` (use logic, not string concat of empty strings)
- The columns in the output sheet must use the ORIGINAL case for `Planned Disposition` (not lowercased). Only use lowercase for comparison.

### 2e – Build Summary sheet
- From Formatted Data, group by `(Warehouse, Carrier)` and sum `Missing Final Event`, `Disposition Mismatch`, `Total Errors`.
- Filter to groups where `Total Errors > 0`.
- Sort by Warehouse ascending, then Carrier ascending.
- Append a Grand Total row: Warehouse=`Grand Total`, Carrier=`-`, sums of the three numeric columns.
- Headers exactly: `Warehouse`, `Carrier`, `Missing Final Events`, `Disposition Mismatches`, `Total Errors`.

### 2f – Write Excel
Write `/root/Returns_Disposition_Audit.xlsx` with three sheets:
- `RawData`: exact copy of the plan table (all original columns, original order).
- `Formatted Data`: the 12-column table built above, with concrete values (no formulas). Ensure integer types for numeric columns (0/1 values).
- `Summary`: the summary table.

Use `openpyxl` engine. Ensure `index=False` for all sheets.

### 2g – Build Word document
Create `/root/Returns_Disposition_Brief.docx` with an executive summary paragraph (3-6 sentences) that includes:
1. Plain-language definition of both checks:
   - "Missing Final Event" means no completed disposition event was recorded for a return line.
   - "Disposition Mismatch" means the final recorded disposition differs from the originally planned disposition.
2. The computed totals: "The audit identified X Missing Final Events, Y Disposition Mismatches, and Z Total Errors."
3. At least one actionable recommendation (e.g., "We recommend investigating root causes at the warehouses with the highest error counts and implementing real-time disposition validation.").
4. Mention at least two specific Return IDs that have the most errors. To find these:
   - Group Formatted Data by `Return ID`, sum `Total Errors`, sort descending, pick top 2+ Return IDs.
   - Include them explicitly like: "High-priority returns requiring immediate review include Return ID XXXX and Return ID YYYY."

Save the document.

### 2h – Validation
After writing both files, reload them and print:
- Sheet names of the Excel file
- Column headers of each sheet
- First 5 rows of `Formatted Data`
- First 5 rows of `Summary`
- Full text content of the Word document
- Value counts for `Missing Final Event`, `Disposition Mismatch`, `Total Errors` columns
- The Grand Total row from Summary

## Step 3 – Verify files exist
Run `ls -la /root/Returns_Disposition_Audit.xlsx /root/Returns_Disposition_Brief.docx` to confirm both files are present and non-empty.

## Critical notes
- Column name matching: inspect actual column names from the source files and map them carefully. Do NOT assume column names match exactly; print them first.
- The alias mapping must be case-insensitive: lowercase both the alias and the final disposition before lookup.
- The `Error Summary` column must contain exact strings: `None`, `Missing Final Event`, `Disposition Mismatch`, or `Missing Final Event, Disposition Mismatch`. No other variations.
- The `Formatted Data` sheet must preserve original `Planned Disposition` text (not lowercased) in the column. Only lowercase for comparison.
- Numeric columns (Missing Final Event, Disposition Mismatch, Total Errors) must be integers (0 or 1), not floats.
- The Word document MUST mention at least two specific Return IDs by their actual ID values from the data. This is a known failure point from similar tasks.

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