# Task Instruction

Execute the following steps in order to produce `/root/Returns_Disposition_Audit.xlsx` and `/root/Returns_Disposition_Brief.docx`.

## Step 0 — Inspect source files

```python
import pandas as pd

plan = pd.read_excel('/root/Return_Plan.xlsx')
print('Return_Plan columns:', list(plan.columns))
print('Return_Plan dtypes:\n', plan.dtypes)
print('Return_Plan shape:', plan.shape)
print(plan.head(10).to_string())
print('---')

events = pd.read_excel('/root/Disposition_Event_Log.xlsx')
print('Event_Log columns:', list(events.columns))
print('Event_Log dtypes:\n', events.dtypes)
print('Event_Log shape:', events.shape)
print(events.head(15).to_string())
print('---')

alias = pd.read_excel('/root/Disposition_Alias.xlsx')
print('Alias columns:', list(alias.columns))
print('Alias dtypes:\n', alias.dtypes)
print('Alias shape:', alias.shape)
print(alias.to_string())
```

Run this first. Read and understand every column name, data type, and sample values before proceeding. Pay special attention to:
- Exact column names (spaces, capitalization) in all three files.
- Data types of `Return ID` and `Line ID` in both the plan and event log (string vs int vs float). They must be cast to the same type before any join.
- The column that holds the event timestamp or sequence number (needed to pick the "latest" completed event).
- The `Event Status` column's exact values (check for leading/trailing spaces, casing).
- The alias file's column names for the alias value and the standard disposition value.
- Any NaN/empty cells in the plan — these must be preserved exactly as they appear in the source (do NOT fill NaN with 'N/A' unless the source literally contains 'N/A').

## Step 1 — Build the processing script

After inspecting the data, write and run a single Python script (`/root/build_audit.py`) that does everything below. Adapt column names to match what you discovered in Step 0.

### 1a) Read data
```python
import pandas as pd
from openpyxl import Workbook
from docx import Document

plan = pd.read_excel('/root/Return_Plan.xlsx')
events = pd.read_excel('/root/Disposition_Event_Log.xlsx')
alias_df = pd.read_excel('/root/Disposition_Alias.xlsx')
```

### 1b) RawData sheet
- Write `plan` exactly as-is (same columns, same order, same values including NaN) to the `RawData` sheet.
- IMPORTANT: Do NOT fill NaN values. However, if the source file literally contains strings like 'N/A', preserve those. Read a few cells manually to confirm.

### 1c) Prepare keys for joining
- Convert `Return ID` and `Line ID` to the same type in both `plan` and `events`. If they are numeric in both, convert to int (after dropping NaN rows in events if any). If they are strings, strip whitespace. The key point: the join must not silently produce zero matches due to type mismatch.
- Verify the join will work by printing the number of matching keys:
```python
plan_keys = set(zip(plan['Return ID'], plan['Line ID']))
event_keys = set(zip(events['Return ID'], events['Line ID']))
print(f'Plan keys: {len(plan_keys)}, Event keys: {len(event_keys)}, Intersection: {len(plan_keys & event_keys)}')
```
If intersection is 0 or unexpectedly small, investigate and fix the type mismatch.

### 1d) Filter events to latest COMPLETED
- Filter events where `Event Status` equals `COMPLETED` (case-insensitive: compare `.str.strip().str.upper()` to `'COMPLETED'`).
- Among those, for each `(Return ID, Line ID)`, keep only the row with the latest timestamp/sequence. Use the timestamp or event-sequence column you identified in Step 0. If there's a tie, keep the last one.
- Result: `latest_events` dataframe with one row per `(Return ID, Line ID)`.

### 1e) Build alias lookup
- From the alias file, build a dictionary mapping `alias_value.strip().lower()` → `standard_disposition.strip()`. Use the actual column names from Step 0.

### 1f) Normalize disposition
- For each row in `latest_events`, take the `Final Disposition` value.
- Look it up (case-insensitive, stripped) in the alias dict. If found, replace with the standard value. Otherwise keep as-is.
- Store the normalized value in a new column `Normalized Disposition`.

### 1g) Left-join and compute flags
- Left-join `plan` with `latest_events` on `(Return ID, Line ID)`. This ensures every plan row appears, and rows without a matching completed event get NaN for event columns.
- Compute:
  - `Missing Final Event` = 1 if `Normalized Disposition` is NaN (i.e., no completed event), else 0. Use `int` type.
  - `Disposition Mismatch` = 1 if `Normalized Disposition` is NOT NaN AND `Normalized Disposition.strip().lower() != Planned Disposition.strip().lower()`, else 0. Use `int` type.
  - `Total Errors` = `Missing Final Event` + `Disposition Mismatch`. Use `int` type.
  - `Error Summary`:
    - If both flags are 0: `'None'`
    - If only Missing: `'Missing Final Event'`
    - If only Mismatch: `'Disposition Mismatch'`
    - If both: `'Missing Final Event, Disposition Mismatch'`

### 1h) Build Formatted Data sheet
- Take the first 8 columns from `plan` (use the exact original column names from the plan file).
- Rename them to exactly: `Return ID`, `Line ID`, `Planned Disposition`, `Reason Code`, `Requested Qty`, `Warehouse`, `Carrier`, `Lane` — but ONLY if the source names differ. If the source already uses these names, keep them.
- Append the 4 computed columns: `Missing Final Event`, `Disposition Mismatch`, `Total Errors`, `Error Summary`.
- Ensure the row order matches `RawData` exactly (same as the original plan order).
- Write concrete values (no formulas). Ensure the flag columns are Python `int` (not float).

### 1i) Build Summary sheet
- From Formatted Data, group by `(Warehouse, Carrier)`.
- For each group, sum `Missing Final Event`, `Disposition Mismatch`, `Total Errors`.
- Keep only groups where `Total Errors > 0`.
- Sort by `Warehouse` ascending, then `Carrier` ascending.
- Append a Grand Total row: `Warehouse='Grand Total'`, `Carrier='-'`, and the dataset-wide sums.
- Column headers exactly: `Warehouse`, `Carrier`, `Missing Final Events`, `Disposition Mismatches`, `Total Errors`.
- Note the plural forms in the summary headers (`Missing Final Events`, `Disposition Mismatches`) vs the singular in Formatted Data (`Missing Final Event`, `Disposition Mismatch`). Use exactly these names.

### 1j) Write Excel
```python
with pd.ExcelWriter('/root/Returns_Disposition_Audit.xlsx', engine='openpyxl') as writer:
    raw_data_df.to_excel(writer, sheet_name='RawData', index=False)
    formatted_df.to_excel(writer, sheet_name='Formatted Data', index=False)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
```

### 1k) Build Word document
- Compute totals: `total_missing`, `total_mismatch`, `total_errors` from the Formatted Data.
- Identify at least 2 Return IDs with the most errors (highest `Total Errors` summed across their lines). These are the "high-priority" return IDs.
- Write 3-6 sentences:
  1. Define `Missing Final Event`: a return line has no completed disposition event in the event log.
  2. Define `Disposition Mismatch`: the final completed disposition does not match the planned disposition.
  3. State the totals: X Missing Final Events, Y Disposition Mismatches, Z Total Errors.
  4. Name at least 2 high-priority Return IDs.
  5. Give an actionable recommendation (e.g., investigate root causes, retrain staff, improve system validations).
- Save to `/root/Returns_Disposition_Brief.docx`.

## Step 2 — Validate outputs

After running the script, verify:
```python
import pandas as pd

# Check sheet names
xl = pd.ExcelFile('/root/Returns_Disposition_Audit.xlsx')
print('Sheets:', xl.sheet_names)
assert xl.sheet_names == ['RawData', 'Formatted Data', 'Summary']

# Check RawData matches source
raw = pd.read_excel('/root/Returns_Disposition_Audit.xlsx', sheet_name='RawData')
plan = pd.read_excel('/root/Return_Plan.xlsx')
assert raw.shape == plan.shape, f'Shape mismatch: {raw.shape} vs {plan.shape}'

# Check Formatted Data
fmt = pd.read_excel('/root/Returns_Disposition_Audit.xlsx', sheet_name='Formatted Data')
print('Formatted Data columns:', list(fmt.columns))
print('Formatted Data shape:', fmt.shape)
print(fmt[['Return ID','Line ID','Missing Final Event','Disposition Mismatch','Total Errors','Error Summary']].to_string())

# Check Summary
smry = pd.read_excel('/root/Returns_Disposition_Audit.xlsx', sheet_name='Summary')
print('Summary columns:', list(smry.columns))
print(smry.to_string())

# Check totals consistency
assert fmt['Total Errors'].sum() == smry['Total Errors'].sum(), 'Total errors mismatch between sheets'
print('Grand Total row:', smry.iloc[-1].to_dict())
assert smry.iloc[-1]['Warehouse'] == 'Grand Total'

# Check Word doc exists and has content
from docx import Document
doc = Document('/root/Returns_Disposition_Brief.docx')
text = ' '.join([p.text for p in doc.paragraphs])
print('Word doc text:', text[:500])
assert str(int(fmt['Total Errors'].sum())) in text, 'Total errors not found in Word doc'
print('All validations passed.')
```

If any assertion fails, diagnose and fix before finishing.

## Critical Reminders
- The #1 failure mode from previous execution was type mismatch on join keys causing zero matches → all Missing Final Event = 0. Print and verify the join intersection count.
- The #2 failure mode was incorrect aggregation. After computing flags, print a few rows to sanity-check before writing.
- Do NOT fill NaN in the plan data with 'N/A' or any other string — preserve the raw data exactly.
- Ensure flag columns are written as integers (0 or 1), not floats (0.0 or 1.0).
- The cross-task hint about NaN→'N/A' is from a DIFFERENT task. Do NOT apply it here unless the source data literally contains 'N/A' strings.

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