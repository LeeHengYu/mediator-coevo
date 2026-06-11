# Task Instruction

## Task: Returns Disposition Audit

You must produce two files:
1. `/root/Returns_Disposition_Audit.xlsx` (3 sheets: `RawData`, `Formatted Data`, `Summary`)
2. `/root/Returns_Disposition_Brief.docx` (executive summary)

### Step 0: Inspect Input Files

Before writing any code, read and print the contents of all three input files to understand their structure, column names, data types, and sample values.

```python
import pandas as pd

plan = pd.read_excel('/root/Return_Plan.xlsx')
print('=== Return_Plan ===')
print(plan.columns.tolist())
print(plan.dtypes)
print(plan.head(20))
print(f'Shape: {plan.shape}')
print()

events = pd.read_excel('/root/Disposition_Event_Log.xlsx')
print('=== Disposition_Event_Log ===')
print(events.columns.tolist())
print(events.dtypes)
print(events.head(20))
print(f'Shape: {events.shape}')
print()

alias = pd.read_excel('/root/Disposition_Alias.xlsx')
print('=== Disposition_Alias ===')
print(alias.columns.tolist())
print(alias.dtypes)
print(alias.head(20))
print(f'Shape: {alias.shape}')
```

Run this first and carefully examine the output. Pay special attention to:
- The exact column names (spaces, capitalization, underscores)
- The data types of `Return ID` and `Line ID` in both `Return_Plan` and `Disposition_Event_Log` — they must match for the join to work
- The exact column name for event status and final disposition in the event log
- The structure of the alias table (which column is the alias, which is the standard name)

### Step 1: Write the Complete Solution Script

After inspecting the data, write a single Python script that does everything. Use the actual column names you observed. Here is the logic blueprint — adapt column names to match what you found:

```python
import pandas as pd
from docx import Document

# 1. Read all files
plan = pd.read_excel('/root/Return_Plan.xlsx')
events = pd.read_excel('/root/Disposition_Event_Log.xlsx')
alias_df = pd.read_excel('/root/Disposition_Alias.xlsx')

# 2. CRITICAL: Ensure Return ID and Line ID have the same type in both dataframes
#    Convert both to string (stripped) to avoid int-vs-string join failures
for col in ['Return ID', 'Line ID']:  # Use actual column names
    plan[col] = plan[col].astype(str).str.strip()
    events[col] = events[col].astype(str).str.strip()

# 3. Filter events to only COMPLETED status (use actual column name for Event Status)
#    Compare case-insensitively
completed = events[events['Event Status'].astype(str).str.strip().str.upper() == 'COMPLETED'].copy()

# 4. Keep only the LATEST completed event per (Return ID, Line ID)
#    Determine which column indicates ordering (timestamp, event ID, row order)
#    If there's a timestamp column, sort by it. Otherwise use the original row index.
#    Sort ascending so that the last row after drop_duplicates(keep='last') is the latest.
completed = completed.sort_index()  # or sort by timestamp if available
completed = completed.drop_duplicates(subset=['Return ID', 'Line ID'], keep='last')

# 5. Build alias mapping (case-insensitive)
#    Inspect alias_df columns — typically: Alias, Standard Disposition (or similar)
#    Build a dict: lowercase alias -> standard disposition
alias_map = dict(zip(
    alias_df.iloc[:, 0].astype(str).str.strip().str.lower(),  # alias column
    alias_df.iloc[:, 1].astype(str).str.strip()  # standard disposition column
))

# 6. Normalize Final Disposition in completed events
completed['Normalized_Disposition'] = completed['Final Disposition'].astype(str).str.strip().str.lower().map(
    lambda x: alias_map.get(x, x)  # if alias found, use standard; else keep raw
)
# Make sure the mapped values are also lowercased for comparison
completed['Normalized_Disposition'] = completed['Normalized_Disposition'].str.lower()

# 7. Left-merge plan with completed events
merged = plan.merge(
    completed[['Return ID', 'Line ID', 'Normalized_Disposition']],
    on=['Return ID', 'Line ID'],
    how='left'
)

# 8. Compute error flags
merged['Missing Final Event'] = merged['Normalized_Disposition'].isna().astype(int)
merged['Disposition Mismatch'] = 0
has_event = merged['Normalized_Disposition'].notna()
merged.loc[has_event, 'Disposition Mismatch'] = (
    merged.loc[has_event, 'Normalized_Disposition'] !=
    merged.loc[has_event, 'Planned Disposition'].astype(str).str.strip().str.lower()
).astype(int)

merged['Total Errors'] = merged['Missing Final Event'] + merged['Disposition Mismatch']

def error_summary(row):
    parts = []
    if row['Missing Final Event'] == 1:
        parts.append('Missing Final Event')
    if row['Disposition Mismatch'] == 1:
        parts.append('Disposition Mismatch')
    return ', '.join(parts) if parts else 'None'

merged['Error Summary'] = merged.apply(error_summary, axis=1)

# 9. Build Formatted Data with exactly 12 columns
formatted_cols = ['Return ID', 'Line ID', 'Planned Disposition', 'Reason Code',
                  'Requested Qty', 'Warehouse', 'Carrier', 'Lane',
                  'Missing Final Event', 'Disposition Mismatch', 'Total Errors', 'Error Summary']
formatted = merged[formatted_cols].copy()

# 10. Verify counts
print(f"Total Missing Final Events: {formatted['Missing Final Event'].sum()}")
print(f"Total Disposition Mismatches: {formatted['Disposition Mismatch'].sum()}")
print(f"Total Errors: {formatted['Total Errors'].sum()}")
print(f"Rows with errors: {(formatted['Total Errors'] > 0).sum()}")

# 11. Build Summary sheet
summary = formatted.groupby(['Warehouse', 'Carrier']).agg(
    **{'Missing Final Events': ('Missing Final Event', 'sum'),
       'Disposition Mismatches': ('Disposition Mismatch', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
).reset_index()
summary = summary[summary['Total Errors'] > 0]
summary = summary.sort_values(['Warehouse', 'Carrier']).reset_index(drop=True)

# Convert aggregated columns to int
for c in ['Missing Final Events', 'Disposition Mismatches', 'Total Errors']:
    summary[c] = summary[c].astype(int)

# Grand Total row
grand = pd.DataFrame([{
    'Warehouse': 'Grand Total',
    'Carrier': '-',
    'Missing Final Events': int(formatted['Missing Final Event'].sum()),
    'Disposition Mismatches': int(formatted['Disposition Mismatch'].sum()),
    'Total Errors': int(formatted['Total Errors'].sum())
}])
summary = pd.concat([summary, grand], ignore_index=True)

# 12. Write Excel
with pd.ExcelWriter('/root/Returns_Disposition_Audit.xlsx', engine='openpyxl') as writer:
    plan.to_excel(writer, sheet_name='RawData', index=False)
    formatted.to_excel(writer, sheet_name='Formatted Data', index=False)
    summary.to_excel(writer, sheet_name='Summary', index=False)

# 13. Write Word document
total_missing = int(formatted['Missing Final Event'].sum())
total_mismatch = int(formatted['Disposition Mismatch'].sum())
total_errors = int(formatted['Total Errors'].sum())

# Find top return IDs by error count
error_by_return = formatted.groupby('Return ID')['Total Errors'].sum().sort_values(ascending=False)
top_returns = error_by_return[error_by_return > 0].head(2).index.tolist()

doc = Document()
doc.add_heading('Returns Disposition Audit – Executive Summary', level=1)

para = (
    f'This audit evaluated returns-processing accuracy using two checks: '
    f'"Missing Final Event" flags any return line lacking a completed disposition event in the event log, '
    f'while "Disposition Mismatch" flags lines where the final recorded disposition differs from the planned disposition. '
    f'Across all audited return lines, we identified {total_missing} Missing Final Events, '
    f'{total_mismatch} Disposition Mismatches, and {total_errors} Total Errors. '
    f'Return IDs {" and ".join(top_returns)} exhibited the most frequent exceptions and should be investigated as high priority. '
    f'We recommend conducting a root-cause analysis on these returns, verifying warehouse scanning procedures, '
    f'and implementing automated disposition-matching alerts to reduce future discrepancies.'
)
doc.add_paragraph(para)
doc.save('/root/Returns_Disposition_Brief.docx')

print('Done.')
```

### CRITICAL NOTES — Read Before Coding

1. **Column name matching**: The blueprint above uses placeholder column names like `'Return ID'`, `'Line ID'`, `'Event Status'`, `'Final Disposition'`, `'Planned Disposition'`. You MUST replace these with the exact column names from your Step 0 inspection. A single mismatched column name will cause the join to silently produce all NaN values, leading to incorrect Missing Final Event counts.

2. **Type alignment for join keys**: The previous execution failed because the merge produced no matches. After inspecting types, cast both `Return ID` and `Line ID` to the same type (string is safest) in BOTH dataframes before merging.

3. **Alias mapping direction**: Carefully check which column in `Disposition_Alias.xlsx` is the alias and which is the standard. The alias column contains variant names that should map TO the standard name. Apply this mapping to the event log's Final Disposition, then compare against the plan's Planned Disposition.

4. **Case-insensitive comparison**: Both the alias lookup and the disposition comparison must be case-insensitive. Lowercase everything before comparing.

5. **Latest event selection**: If there's a timestamp or sequence column, sort by it before keeping the last duplicate. If not, use the original row order (sort by index).

6. **Verification**: After computing the formatted data, print the total counts and a few sample rows with errors to verify the logic before writing files. The expected total errors should be 12 based on the verifier feedback.

7. **Integer values**: Ensure `Missing Final Event`, `Disposition Mismatch`, and `Total Errors` are written as integers (not floats) in the Excel file.

### Step 2: Validate Output

After running the script, verify:
```python
import pandas as pd
result = pd.read_excel('/root/Returns_Disposition_Audit.xlsx', sheet_name=None)
print('Sheets:', list(result.keys()))
print('RawData shape:', result['RawData'].shape)
print('Formatted Data shape:', result['Formatted Data'].shape)
print('Formatted Data columns:', result['Formatted Data'].columns.tolist())
print('Summary:\n', result['Summary'])
print('Total errors:', result['Formatted Data']['Total Errors'].sum())
print('Missing events:', result['Formatted Data']['Missing Final Event'].sum())
print('Mismatches:', result['Formatted Data']['Disposition Mismatch'].sum())

from docx import Document
doc = Document('/root/Returns_Disposition_Brief.docx')
for p in doc.paragraphs:
    print(p.text)
```

Confirm that:
- Sheet names are exactly `RawData`, `Formatted Data`, `Summary`
- Formatted Data has exactly 12 columns with the specified headers
- Summary only includes groups with Total Errors > 0, plus the Grand Total row
- The Word document mentions the numeric totals and at least two return IDs
- Total Errors = 12 (based on verifier feedback)

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