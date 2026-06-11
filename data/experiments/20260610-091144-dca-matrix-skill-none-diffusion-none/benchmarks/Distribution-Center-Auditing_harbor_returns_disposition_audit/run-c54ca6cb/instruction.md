# Task Instruction

Execute the following Python script in a single step to produce both deliverables. Before running, install any missing packages (`pip install openpyxl python-docx pandas`). Then run the script below.

```python
import pandas as pd
from openpyxl import load_workbook, Workbook
from docx import Document
import os

# ── 1. Load source data ──────────────────────────────────────────────
plan_df = pd.read_excel('/root/Return_Plan.xlsx')
event_df = pd.read_excel('/root/Disposition_Event_Log.xlsx')
alias_df = pd.read_excel('/root/Disposition_Alias.xlsx')

# ── 2. Inspect columns (debug print) ────────────────────────────────
print('Plan columns:', list(plan_df.columns))
print('Event columns:', list(event_df.columns))
print('Alias columns:', list(alias_df.columns))
print('Plan shape:', plan_df.shape)
print('Event shape:', event_df.shape)
print('Alias shape:', alias_df.shape)
print()
print('Plan head:')
print(plan_df.head(10).to_string())
print()
print('Event head:')
print(event_df.head(10).to_string())
print()
print('Alias head:')
print(alias_df.head(10).to_string())
print()

# Print unique Event Status values for debugging
print('Unique Event Status values:', event_df['Event Status'].unique() if 'Event Status' in event_df.columns else 'COLUMN NOT FOUND')

# ── 3. Normalize join keys to strings ────────────────────────────────
plan_df['Return ID'] = plan_df['Return ID'].astype(str).str.strip()
plan_df['Line ID'] = plan_df['Line ID'].astype(str).str.strip()
event_df['Return ID'] = event_df['Return ID'].astype(str).str.strip()
event_df['Line ID'] = event_df['Line ID'].astype(str).str.strip()

# ── 4. Filter events: keep only COMPLETED (case-insensitive) ────────
event_df['_status_upper'] = event_df['Event Status'].astype(str).str.strip().str.upper()
completed_events = event_df[event_df['_status_upper'] == 'COMPLETED'].copy()

print(f'\nCompleted events count: {len(completed_events)}')
print(completed_events.head(10).to_string())

# ── 5. Keep only the LATEST completed event per (Return ID, Line ID) ─
# Determine the timestamp/sequence column for ordering
# Try common names
for col in completed_events.columns:
    print(f'  col: {col}  dtype: {completed_events[col].dtype}')

# Identify a suitable ordering column (Event Timestamp, Event Date, Event Seq, etc.)
order_col = None
for candidate in ['Event Timestamp', 'Event Date', 'Event Time', 'Event Seq', 'Sequence', 'Timestamp', 'Date']:
    if candidate in completed_events.columns:
        order_col = candidate
        break

if order_col is None:
    # If there's no obvious ordering column, use the dataframe index (row order)
    print('WARNING: No ordering column found; using original row order as proxy.')
    completed_events = completed_events.reset_index(drop=False).rename(columns={'index': '_orig_idx'})
    order_col = '_orig_idx'
else:
    print(f'Using ordering column: {order_col}')

completed_events = completed_events.sort_values(order_col)
latest_events = completed_events.groupby(['Return ID', 'Line ID'], as_index=False).last()

print(f'\nLatest completed events count: {len(latest_events)}')
print(latest_events.head(10).to_string())

# ── 6. Build alias lookup (case-insensitive) ────────────────────────
# Alias file maps alias -> standard disposition
# Identify the alias and standard columns
print('\nAlias columns:', list(alias_df.columns))
print(alias_df.to_string())

# Expect columns like: Alias, Standard Disposition (or similar)
alias_cols = list(alias_df.columns)
# Heuristic: first column is alias, second is standard
alias_col_name = alias_cols[0]
standard_col_name = alias_cols[1]
print(f'Alias column: {alias_col_name}, Standard column: {standard_col_name}')

alias_map = {}
for _, row in alias_df.iterrows():
    key = str(row[alias_col_name]).strip().upper()
    val = str(row[standard_col_name]).strip()
    alias_map[key] = val

print('Alias map:', alias_map)

# ── 7. Identify Final Disposition column in event log ────────────────
final_disp_col = None
for candidate in ['Final Disposition', 'Disposition', 'Final_Disposition']:
    if candidate in latest_events.columns:
        final_disp_col = candidate
        break
print(f'Final Disposition column in events: {final_disp_col}')

# ── 8. Merge plan with latest events ────────────────────────────────
merged = plan_df.merge(
    latest_events[['Return ID', 'Line ID', final_disp_col]],
    on=['Return ID', 'Line ID'],
    how='left'
)

print(f'\nMerged shape: {merged.shape}')
print(merged.head(10).to_string())

# ── 9. Normalize Final Disposition using alias map ───────────────────
def normalize_disposition(raw):
    if pd.isna(raw):
        return None
    raw_str = str(raw).strip()
    upper = raw_str.upper()
    if upper in alias_map:
        return alias_map[upper]
    return raw_str

merged['_normalized_final'] = merged[final_disp_col].apply(normalize_disposition)

# ── 10. Compute error columns ────────────────────────────────────────
merged['Missing Final Event'] = merged[final_disp_col].isna().astype(int)

def check_mismatch(row):
    if pd.isna(row[final_disp_col]):
        return 0  # No event → mismatch not applicable
    planned = str(row['Planned Disposition']).strip().upper()
    normalized = str(row['_normalized_final']).strip().upper()
    return 0 if planned == normalized else 1

merged['Disposition Mismatch'] = merged.apply(check_mismatch, axis=1)
merged['Total Errors'] = merged['Missing Final Event'] + merged['Disposition Mismatch']

def error_summary(row):
    m = row['Missing Final Event']
    d = row['Disposition Mismatch']
    if m == 1 and d == 1:
        return 'Missing Final Event, Disposition Mismatch'
    elif m == 1:
        return 'Missing Final Event'
    elif d == 1:
        return 'Disposition Mismatch'
    else:
        return 'None'

merged['Error Summary'] = merged.apply(error_summary, axis=1)

# ── 11. Prepare Formatted Data (first 8 plan cols + 4 new cols) ──────
first_8 = list(plan_df.columns[:8])
new_4 = ['Missing Final Event', 'Disposition Mismatch', 'Total Errors', 'Error Summary']
formatted_df = merged[first_8 + new_4].copy()

# Ensure the first 8 column headers match spec exactly
expected_headers = ['Return ID', 'Line ID', 'Planned Disposition', 'Reason Code',
                    'Requested Qty', 'Warehouse', 'Carrier', 'Lane']
formatted_df.columns = expected_headers + new_4

print('\nFormatted Data head:')
print(formatted_df.head(20).to_string())
print(f'\nTotal Missing Final Events: {formatted_df["Missing Final Event"].sum()}')
print(f'Total Disposition Mismatches: {formatted_df["Disposition Mismatch"].sum()}')
print(f'Total Errors: {formatted_df["Total Errors"].sum()}')

# ── 12. Build Summary sheet ──────────────────────────────────────────
summary_agg = formatted_df.groupby(['Warehouse', 'Carrier'], as_index=False).agg(
    **{'Missing Final Events': ('Missing Final Event', 'sum'),
       'Disposition Mismatches': ('Disposition Mismatch', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
)

# Keep only groups with Total Errors > 0
summary_agg = summary_agg[summary_agg['Total Errors'] > 0].copy()
summary_agg = summary_agg.sort_values(['Warehouse', 'Carrier']).reset_index(drop=True)

# Grand Total row
grand_total = pd.DataFrame([{
    'Warehouse': 'Grand Total',
    'Carrier': '-',
    'Missing Final Events': formatted_df['Missing Final Event'].sum(),
    'Disposition Mismatches': formatted_df['Disposition Mismatch'].sum(),
    'Total Errors': formatted_df['Total Errors'].sum()
}])
summary_df = pd.concat([summary_agg, grand_total], ignore_index=True)

# Ensure numeric columns are int
for c in ['Missing Final Events', 'Disposition Mismatches', 'Total Errors']:
    summary_df[c] = summary_df[c].astype(int)

print('\nSummary:')
print(summary_df.to_string())

# ── 13. Write Excel workbook ─────────────────────────────────────────
outpath = '/root/Returns_Disposition_Audit.xlsx'
with pd.ExcelWriter(outpath, engine='openpyxl') as writer:
    plan_df.to_excel(writer, sheet_name='RawData', index=False)
    formatted_df.to_excel(writer, sheet_name='Formatted Data', index=False)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print(f'\nWrote {outpath}')

# ── 14. Identify top Return IDs by error count ──────────────────────
error_by_return = formatted_df.groupby('Return ID')['Total Errors'].sum()
error_by_return = error_by_return[error_by_return > 0].sort_values(ascending=False)
top_returns = list(error_by_return.head(2).index)
print(f'Top error Return IDs: {top_returns}')

total_missing = int(formatted_df['Missing Final Event'].sum())
total_mismatch = int(formatted_df['Disposition Mismatch'].sum())
total_errors = int(formatted_df['Total Errors'].sum())

# ── 15. Write Word brief ─────────────────────────────────────────────
doc = Document()
doc.add_heading('Returns Disposition Audit – Executive Summary', level=1)

para_text = (
    f'This audit evaluated the accuracy of returns disposition processing across all warehouses. '
    f'Two checks were applied to every return line: (1) "Missing Final Event" flags lines where no '
    f'completed disposition event was recorded in the event log, meaning the return was never finalized; '
    f'(2) "Disposition Mismatch" flags lines where the final recorded disposition differs from the '
    f'planned disposition, indicating the item was routed or processed incorrectly. '
    f'Across the dataset, the audit identified {total_missing} Missing Final Events, '
    f'{total_mismatch} Disposition Mismatches, and {total_errors} Total Errors. '
    f'High-priority return IDs with frequent exceptions include {top_returns[0]}'
)
if len(top_returns) > 1:
    para_text += f' and {top_returns[1]}'
para_text += (
    f', which should be investigated first. '
    f'We recommend implementing real-time event-completion alerts and periodic disposition-code '
    f'reconciliation to reduce these discrepancies going forward.'
)

doc.add_paragraph(para_text)

wordpath = '/root/Returns_Disposition_Brief.docx'
doc.save(wordpath)
print(f'Wrote {wordpath}')
print('\nDone.')
```

After running the script, verify:
1. `/root/Returns_Disposition_Audit.xlsx` exists and has exactly three sheets: `RawData`, `Formatted Data`, `Summary`.
2. `/root/Returns_Disposition_Brief.docx` exists.
3. Review the printed debug output carefully. If the script fails or produces unexpected results (e.g., zero errors when errors are expected), diagnose using the printed column names, data samples, and unique values. Common pitfalls to watch for:
   - Column names in the source files may differ from expected (extra spaces, different casing). Adapt the script accordingly.
   - `Return ID` / `Line ID` type mismatches between Plan and Event Log (int vs string).
   - `Event Status` values may have trailing whitespace or different casing.
   - The ordering column for 'latest event' may have a non-obvious name.
   - The alias file column names may differ from assumed names.

If the script fails or produces clearly wrong results, inspect the source files' actual column names and a few rows, then fix and re-run.

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