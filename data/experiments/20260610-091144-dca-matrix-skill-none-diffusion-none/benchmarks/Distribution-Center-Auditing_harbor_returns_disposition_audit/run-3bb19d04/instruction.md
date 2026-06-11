# Task Instruction

Execute the following Python script to produce both deliverables. Before running, inspect the three source files to understand their structure.

```bash
cd /root
python3 << 'PYEOF'
import pandas as pd
import openpyxl
from docx import Document

# ── 1. Read source files ──────────────────────────────────────────────────
plan = pd.read_excel('Return_Plan.xlsx')
events = pd.read_excel('Disposition_Event_Log.xlsx')
alias_df = pd.read_excel('Disposition_Alias.xlsx')

print('=== Plan columns:', list(plan.columns))
print('Plan shape:', plan.shape)
print(plan.head())
print('\n=== Events columns:', list(events.columns))
print('Events shape:', events.shape)
print(events.head(10))
print('\n=== Alias columns:', list(alias_df.columns))
print('Alias shape:', alias_df.shape)
print(alias_df.head(10))

# ── 2. Build alias lookup (case-insensitive) ─────────────────────────────
# Alias file maps variant names → standard disposition
# Identify the alias column and standard column
alias_cols = list(alias_df.columns)
print('\nAlias columns:', alias_cols)
print(alias_df)

# Build dict: lowercase alias -> standard disposition
alias_map = {}
for _, row in alias_df.iterrows():
    # Try to figure out which column is the alias and which is the standard
    # Typically: Alias, Standard Disposition (or similar)
    alias_val = str(row.iloc[0]).strip()
    standard_val = str(row.iloc[1]).strip()
    alias_map[alias_val.lower()] = standard_val

print('\nAlias map:', alias_map)

# ── 3. Filter events to COMPLETED only, keep latest per (Return ID, Line ID)
event_status_col = [c for c in events.columns if 'status' in c.lower()]
print('\nEvent status column candidates:', event_status_col)

# Identify the status column
status_col = event_status_col[0] if event_status_col else None
print(f'Using status column: {status_col}')
print('Unique statuses:', events[status_col].unique() if status_col else 'N/A')

# Filter to COMPLETED
completed = events[events[status_col].astype(str).str.strip().str.upper() == 'COMPLETED'].copy()
print(f'\nCompleted events: {len(completed)}')

# Identify Return ID and Line ID columns in events
ret_id_col_ev = [c for c in events.columns if 'return' in c.lower() and 'id' in c.lower()][0]
line_id_col_ev = [c for c in events.columns if 'line' in c.lower() and 'id' in c.lower()][0]
print(f'Event Return ID col: {ret_id_col_ev}, Line ID col: {line_id_col_ev}')

# Identify timestamp/sequence column to determine "latest"
# Look for date, timestamp, or event columns
date_candidates = [c for c in events.columns if any(k in c.lower() for k in ['date', 'time', 'stamp', 'seq', 'event_id', 'log'])]
print(f'Date/sequence candidates: {date_candidates}')
print(completed.dtypes)

# Sort by all date candidates descending then take last (or first after desc sort)
# We need to find the right column to sort by for "latest"
# Let's check all columns
for c in events.columns:
    print(f'  {c}: dtype={events[c].dtype}, sample={events[c].iloc[0] if len(events)>0 else "N/A"}')

PYEOF
```

After inspecting the output, run the full processing script:

```bash
cd /root
python3 << 'PYEOF'
import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from docx import Document
import numpy as np

# ── 1. Read source files ──────────────────────────────────────────────────
plan = pd.read_excel('Return_Plan.xlsx')
events = pd.read_excel('Disposition_Event_Log.xlsx')
alias_df = pd.read_excel('Disposition_Alias.xlsx')

# ── 2. Build alias lookup ─────────────────────────────────────────────────
alias_map = {}
for _, row in alias_df.iterrows():
    alias_val = str(row.iloc[0]).strip().lower()
    standard_val = str(row.iloc[1]).strip()
    alias_map[alias_val] = standard_val

print('Alias map:', alias_map)

# ── 3. Identify columns ──────────────────────────────────────────────────
# Plan columns
plan_cols = list(plan.columns)
print('Plan columns:', plan_cols)

# The first 8 columns of the plan should map to our required headers
# Required: Return ID, Line ID, Planned Disposition, Reason Code, Requested Qty, Warehouse, Carrier, Lane
required_headers = ['Return ID', 'Line ID', 'Planned Disposition', 'Reason Code',
                    'Requested Qty', 'Warehouse', 'Carrier', 'Lane']

# Map plan columns to required headers (they should already match or be close)
# Use the plan as-is for RawData; for Formatted Data use first 8 cols

# Event columns
ev_cols = list(events.columns)
print('Event columns:', ev_cols)

# Find key columns in events
ret_id_col = [c for c in ev_cols if 'return' in c.lower() and 'id' in c.lower()][0]
line_id_col = [c for c in ev_cols if 'line' in c.lower() and 'id' in c.lower()][0]
status_col = [c for c in ev_cols if 'status' in c.lower()][0]
final_disp_col = [c for c in ev_cols if 'disposition' in c.lower() or 'final' in c.lower()]
print('Final disposition candidates:', final_disp_col)
# Pick the one with 'final' and 'disposition' or just 'disposition'
final_disp_col = [c for c in ev_cols if 'final' in c.lower() and 'disp' in c.lower()]
if not final_disp_col:
    final_disp_col = [c for c in ev_cols if 'disp' in c.lower()]
final_disp_col = final_disp_col[0]
print(f'Using: ret={ret_id_col}, line={line_id_col}, status={status_col}, disp={final_disp_col}')

# Find a timestamp/sequence column for "latest"
time_cols = [c for c in ev_cols if any(k in c.lower() for k in ['date', 'time', 'stamp', 'seq', 'created', 'updated', 'log'])]
print('Time/seq candidates:', time_cols)
# If no obvious time column, use the index (last row = latest)
# Also check for Event ID or similar
if not time_cols:
    time_cols = [c for c in ev_cols if 'event' in c.lower() and 'id' in c.lower()]
    print('Fallback to event ID:', time_cols)

# ── 4. Filter COMPLETED events, keep latest per (Return ID, Line ID) ─────
completed = events[events[status_col].astype(str).str.strip().str.upper() == 'COMPLETED'].copy()
print(f'Total events: {len(events)}, Completed: {len(completed)}')

# Sort by time column (or index) to identify latest
if time_cols:
    sort_col = time_cols[0]
    completed = completed.sort_values(sort_col, ascending=True)
else:
    sort_col = None
    # Use original index order; last occurrence = latest

# Keep latest: last occurrence per (Return ID, Line ID) after sorting
latest = completed.drop_duplicates(subset=[ret_id_col, line_id_col], keep='last').copy()
print(f'Latest completed events: {len(latest)}')

# ── 5. Normalize dispositions ─────────────────────────────────────────────
def normalize_disposition(raw_disp):
    """Map raw disposition through alias table, return normalized string."""
    if pd.isna(raw_disp):
        return ''
    raw_str = str(raw_disp).strip()
    raw_lower = raw_str.lower()
    if raw_lower in alias_map:
        return alias_map[raw_lower]
    return raw_str

latest['_normalized_disp'] = latest[final_disp_col].apply(normalize_disposition)
print('\nSample latest events:')
print(latest[[ret_id_col, line_id_col, final_disp_col, '_normalized_disp']].head(10))

# ── 6. Build Formatted Data ───────────────────────────────────────────────
# Plan Return ID and Line ID columns
plan_ret_col = [c for c in plan_cols if 'return' in c.lower() and 'id' in c.lower()][0]
plan_line_col = [c for c in plan_cols if 'line' in c.lower() and 'id' in c.lower()][0]
plan_disp_col = [c for c in plan_cols if 'planned' in c.lower() and 'disp' in c.lower()]
if not plan_disp_col:
    plan_disp_col = [c for c in plan_cols if 'disp' in c.lower()]
plan_disp_col = plan_disp_col[0]
print(f'Plan cols: ret={plan_ret_col}, line={plan_line_col}, disp={plan_disp_col}')

# Merge plan with latest events
merged = plan.merge(
    latest[[ret_id_col, line_id_col, '_normalized_disp']],
    left_on=[plan_ret_col, plan_line_col],
    right_on=[ret_id_col, line_id_col],
    how='left',
    suffixes=('', '_ev')
)

# Drop duplicate join key columns if created
for c in [ret_id_col + '_ev', line_id_col + '_ev']:
    if c in merged.columns:
        merged.drop(columns=[c], inplace=True)

print(f'Merged rows: {len(merged)}, Plan rows: {len(plan)}')

# Compute error columns
merged['Missing Final Event'] = merged['_normalized_disp'].apply(
    lambda x: 1 if (pd.isna(x) or x == '') else 0
)

def check_mismatch(row):
    if row['Missing Final Event'] == 1:
        return 0
    planned = str(row[plan_disp_col]).strip().lower()
    actual = str(row['_normalized_disp']).strip().lower()
    # Also normalize the planned disposition through alias map
    planned_normalized = alias_map.get(planned, str(row[plan_disp_col]).strip()).lower()
    return 0 if planned_normalized == actual.lower() else 1

merged['Disposition Mismatch'] = merged.apply(check_mismatch, axis=1)
merged['Total Errors'] = merged['Missing Final Event'] + merged['Disposition Mismatch']

def error_summary(row):
    missing = row['Missing Final Event']
    mismatch = row['Disposition Mismatch']
    if missing and mismatch:
        return 'Missing Final Event, Disposition Mismatch'
    elif missing:
        return 'Missing Final Event'
    elif mismatch:
        return 'Disposition Mismatch'
    else:
        return 'None'

merged['Error Summary'] = merged.apply(error_summary, axis=1)

# Select first 8 plan columns + 4 new columns
first_8 = plan_cols[:8]
formatted_cols = first_8 + ['Missing Final Event', 'Disposition Mismatch', 'Total Errors', 'Error Summary']

# Rename first 8 columns to required headers if needed
rename_map = dict(zip(first_8, required_headers))
formatted = merged[first_8 + ['Missing Final Event', 'Disposition Mismatch', 'Total Errors', 'Error Summary']].copy()
formatted.rename(columns=rename_map, inplace=True)

print('\nFormatted Data sample:')
print(formatted.head(10))
print(f'\nTotal Missing Final Events: {formatted["Missing Final Event"].sum()}')
print(f'Total Disposition Mismatches: {formatted["Disposition Mismatch"].sum()}')
print(f'Total Errors: {formatted["Total Errors"].sum()}')

# ── 7. Build Summary ──────────────────────────────────────────────────────
# Aggregate by (Warehouse, Carrier) where Total Errors > 0
wh_col = 'Warehouse'
carrier_col = 'Carrier'

grouped = formatted.groupby([wh_col, carrier_col]).agg(
    **{'Missing Final Events': ('Missing Final Event', 'sum'),
       'Disposition Mismatches': ('Disposition Mismatch', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
).reset_index()

# Filter to groups with Total Errors > 0
summary = grouped[grouped['Total Errors'] > 0].copy()
summary = summary.sort_values([wh_col, carrier_col], ascending=[True, True]).reset_index(drop=True)

# Grand Total row
grand = pd.DataFrame([{
    wh_col: 'Grand Total',
    carrier_col: '-',
    'Missing Final Events': formatted['Missing Final Event'].sum(),
    'Disposition Mismatches': formatted['Disposition Mismatch'].sum(),
    'Total Errors': formatted['Total Errors'].sum()
}])
summary = pd.concat([summary, grand], ignore_index=True)

# Ensure integer types
for c in ['Missing Final Events', 'Disposition Mismatches', 'Total Errors']:
    summary[c] = summary[c].astype(int)

print('\nSummary:')
print(summary)

# ── 8. Write Excel ────────────────────────────────────────────────────────
with pd.ExcelWriter('Returns_Disposition_Audit.xlsx', engine='openpyxl') as writer:
    # RawData - exact copy of plan
    plan.to_excel(writer, sheet_name='RawData', index=False)
    # Formatted Data
    formatted.to_excel(writer, sheet_name='Formatted Data', index=False)
    # Summary
    summary.to_excel(writer, sheet_name='Summary', index=False)

print('\nExcel written successfully.')

# ── 9. Verify Excel ──────────────────────────────────────────────────────
verify = pd.read_excel('Returns_Disposition_Audit.xlsx', sheet_name=None)
print('Sheets:', list(verify.keys()))
for s, df in verify.items():
    print(f'  {s}: {df.shape}, cols={list(df.columns)}')

# ── 10. Word Brief ────────────────────────────────────────────────────────
total_missing = int(formatted['Missing Final Event'].sum())
total_mismatch = int(formatted['Disposition Mismatch'].sum())
total_errors = int(formatted['Total Errors'].sum())

# Find top return IDs with most errors
error_by_return = formatted.groupby('Return ID')['Total Errors'].sum()
top_returns = error_by_return.sort_values(ascending=False)
top_2 = list(top_returns[top_returns > 0].head(2).index)
print(f'\nTop 2 return IDs: {top_2}')

doc = Document()
doc.add_heading('Returns Disposition Audit – Executive Summary', level=1)

brief = (
    f'This audit examined returns disposition accuracy using two key checks: '
    f'"Missing Final Event" flags return lines with no completed disposition event recorded in the system, '
    f'while "Disposition Mismatch" identifies lines where the final completed disposition differs from the planned disposition. '
    f'Across the audited dataset, there were {total_missing} Missing Final Events, '
    f'{total_mismatch} Disposition Mismatches, and {total_errors} Total Errors. '
    f'High-priority return IDs requiring immediate attention include {top_2[0]} and {top_2[1]}, '
    f'which had the most frequent exceptions. '
    f'We recommend implementing automated disposition confirmation alerts at the warehouse level '
    f'and conducting targeted retraining for carriers associated with the highest error rates '
    f'to reduce both missing events and mismatches going forward.'
)

doc.add_paragraph(brief)
doc.save('Returns_Disposition_Brief.docx')
print('Word brief written successfully.')
print('\nDone.')
PYEOF
```

After running, verify the outputs:
```bash
python3 -c "
import pandas as pd
from docx import Document

# Verify Excel
xl = pd.read_excel('/root/Returns_Disposition_Audit.xlsx', sheet_name=None)
for name, df in xl.items():
    print(f'Sheet: {name}')
    print(f'  Shape: {df.shape}')
    print(f'  Columns: {list(df.columns)}')
    print(df.head())
    print()

# Verify Word
doc = Document('/root/Returns_Disposition_Brief.docx')
for p in doc.paragraphs:
    print(p.text)
"
```

IMPORTANT NOTES:
1. Run the FIRST script block (inspection) first. Examine its output carefully to understand the exact column names and data types in all three source files.
2. If the column names in the source files don't match what the second script expects, ADAPT the second script accordingly before running it.
3. Pay special attention to:
   - The exact column names in Disposition_Event_Log.xlsx (Return ID, Line ID, Event Status, Final Disposition, and any timestamp/sequence column)
   - The exact column names in Disposition_Alias.xlsx (which column is the alias, which is the standard)
   - The exact column names in Return_Plan.xlsx (the first 8 columns)
4. When normalizing dispositions for comparison, normalize BOTH the planned disposition AND the final disposition through the alias map, then compare case-insensitively.
5. The merge must be a LEFT join from the plan to events, so every plan row appears in the output.
6. For the "latest" COMPLETED event: sort by whatever timestamp/date/sequence column exists and keep the last one per (Return ID, Line ID).
7. Ensure numeric columns in the output (Missing Final Event, Disposition Mismatch, Total Errors) contain plain integers, not floats.
8. The Error Summary for a row where Missing Final Event=1 cannot also have Disposition Mismatch=1 (since mismatch is only checked when an event exists). However, preserve the logic as specified.
9. Verify that the total errors in the Word document match the actual computed totals from the Formatted Data sheet.

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