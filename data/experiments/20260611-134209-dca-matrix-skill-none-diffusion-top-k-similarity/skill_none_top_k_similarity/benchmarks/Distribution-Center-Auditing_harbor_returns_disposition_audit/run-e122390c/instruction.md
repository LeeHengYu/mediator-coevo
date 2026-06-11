# Task Instruction

Execute the following Python script to produce both deliverables. Before writing the script, inspect the three input files to understand their exact column names, data types, and content.

## Step 1: Inspect input files
```bash
cd /root
python3 -c "
import openpyxl

for fname in ['Return_Plan.xlsx', 'Disposition_Event_Log.xlsx', 'Disposition_Alias.xlsx']:
    wb = openpyxl.load_workbook(fname)
    for sn in wb.sheetnames:
        ws = wb[sn]
        print(f'=== {fname} / {sn} ===')
        print(f'Rows: {ws.max_row}, Cols: {ws.max_column}')
        for r in range(1, min(ws.max_row+1, 8)):
            print([ws.cell(r, c).value for c in range(1, ws.max_column+1)])
    wb.close()
"
```

Carefully note:
- The exact column headers in each file (spelling, casing)
- The column used for timestamps or ordering in the event log (look for a date/time column, or an Event ID, or row order)
- The structure of Disposition_Alias.xlsx (which column is the alias, which is the standard name)

## Step 2: Write and run the main script

After inspecting, write `/root/solve.py` with the following logic:

```python
import pandas as pd
from openpyxl import load_workbook, Workbook
from docx import Document

# ---- Load data ----
plan = pd.read_excel('Return_Plan.xlsx')
events = pd.read_excel('Disposition_Event_Log.xlsx')
alias_df = pd.read_excel('Disposition_Alias.xlsx')

# Print columns to verify
print('Plan columns:', list(plan.columns))
print('Events columns:', list(events.columns))
print('Alias columns:', list(alias_df.columns))
print('Plan shape:', plan.shape)
print('Events shape:', events.shape)
print('Alias shape:', alias_df.shape)
print('Events sample:')
print(events.head(10).to_string())
print('Alias data:')
print(alias_df.to_string())

# ---- Build alias mapping (case-insensitive) ----
# Determine which column is the alias and which is the standard.
# Typically: Alias -> Standard Disposition
alias_map = {}
for _, row in alias_df.iterrows():
    # Adjust column names after inspection
    alias_val = str(row.iloc[0]).strip()
    standard_val = str(row.iloc[1]).strip()
    alias_map[alias_val.lower()] = standard_val

print('Alias map:', alias_map)

# ---- Filter events: keep only COMPLETED status ----
# Adjust 'Event Status' column name after inspection
status_col = [c for c in events.columns if 'status' in c.lower()][0]
print(f'Using status column: {status_col}')

completed = events[events[status_col].astype(str).str.strip().str.upper() == 'COMPLETED'].copy()
print(f'Completed events: {len(completed)}')

# ---- Keep latest COMPLETED event per (Return ID, Line ID) ----
# Determine ordering column: look for timestamp, event date, event id, etc.
# Use the original DataFrame index as a fallback for 'latest' (last row = latest)
completed = completed.copy()
completed['_orig_idx'] = completed.index  # preserve original row order

# Check for date/time columns
date_cols = [c for c in completed.columns if any(kw in c.lower() for kw in ['date', 'time', 'timestamp'])]
print(f'Potential date columns: {date_cols}')

# Identify Return ID and Line ID columns
return_id_col = [c for c in events.columns if 'return' in c.lower() and 'id' in c.lower()][0]
line_id_col = [c for c in events.columns if 'line' in c.lower() and 'id' in c.lower()][0]
final_disp_col = [c for c in events.columns if 'final' in c.lower() or 'disposition' in c.lower()]
print(f'Return ID col: {return_id_col}, Line ID col: {line_id_col}')
print(f'Potential final disposition cols: {final_disp_col}')

# Pick the disposition column from events - the one with 'Final Disposition' or similar
disp_col = [c for c in events.columns if 'disposition' in c.lower()]
print(f'Disposition columns in events: {disp_col}')
# Use the most specific one
final_disp_col_name = disp_col[0]  # adjust after inspection

if date_cols:
    sort_col = date_cols[0]
    completed = completed.sort_values(sort_col)
else:
    sort_col = '_orig_idx'
    completed = completed.sort_values('_orig_idx')

print(f'Sorting completed events by: {sort_col}')

# Keep last (latest) per group
latest = completed.groupby([return_id_col, line_id_col]).last().reset_index()
print(f'Latest completed events: {len(latest)}')

# ---- Build lookup: (Return ID, Line ID) -> normalized final disposition ----
def normalize_disposition(raw_disp):
    raw = str(raw_disp).strip()
    raw_lower = raw.lower()
    if raw_lower in alias_map:
        return alias_map[raw_lower]
    return raw

event_lookup = {}
for _, row in latest.iterrows():
    key = (row[return_id_col], row[line_id_col])
    event_lookup[key] = normalize_disposition(row[final_disp_col_name])

print(f'Event lookup size: {len(event_lookup)}')

# ---- Build Formatted Data ----
# Plan columns - first 8
plan_cols = list(plan.columns[:8])
print(f'Plan first 8 cols: {plan_cols}')

# Map plan column names to standard names
plan_return_id = plan_cols[0]  # Return ID
plan_line_id = plan_cols[1]    # Line ID  
plan_planned_disp = plan_cols[2]  # Planned Disposition

formatted = plan.copy()

missing_list = []
mismatch_list = []
total_errors_list = []
summary_list = []

for idx, row in formatted.iterrows():
    key = (row[plan_return_id], row[plan_line_id])
    
    if key not in event_lookup:
        missing = 1
        mismatch = 0  # Can't have mismatch if no event
    else:
        missing = 0
        norm_final = event_lookup[key]
        planned = str(row[plan_planned_disp]).strip()
        if norm_final.lower() == planned.lower():
            mismatch = 0
        else:
            mismatch = 1
    
    total = missing + mismatch
    
    if missing == 1 and mismatch == 1:
        summary_text = 'Missing Final Event, Disposition Mismatch'
    elif missing == 1:
        summary_text = 'Missing Final Event'
    elif mismatch == 1:
        summary_text = 'Disposition Mismatch'
    else:
        summary_text = 'None'
    
    missing_list.append(missing)
    mismatch_list.append(mismatch)
    total_errors_list.append(total)
    summary_list.append(summary_text)

formatted['Missing Final Event'] = missing_list
formatted['Disposition Mismatch'] = mismatch_list
formatted['Total Errors'] = total_errors_list
formatted['Error Summary'] = summary_list

print(f'\nError totals: Missing={sum(missing_list)}, Mismatch={sum(mismatch_list)}, Total={sum(total_errors_list)}')

# ---- Build Summary ----
wh_col = plan_cols[5]  # Warehouse
carrier_col = plan_cols[6]  # Carrier

error_rows = formatted[formatted['Total Errors'] > 0]
if len(error_rows) > 0:
    summary_df = error_rows.groupby([wh_col, carrier_col]).agg(
        **{'Missing Final Events': ('Missing Final Event', 'sum'),
           'Disposition Mismatches': ('Disposition Mismatch', 'sum'),
           'Total Errors': ('Total Errors', 'sum')}
    ).reset_index()
    summary_df = summary_df.sort_values([wh_col, carrier_col]).reset_index(drop=True)
else:
    summary_df = pd.DataFrame(columns=[wh_col, carrier_col, 'Missing Final Events', 'Disposition Mismatches', 'Total Errors'])

# Rename columns to exact required names
summary_df = summary_df.rename(columns={wh_col: 'Warehouse', carrier_col: 'Carrier'})

# Grand Total row
grand = pd.DataFrame([{
    'Warehouse': 'Grand Total',
    'Carrier': '-',
    'Missing Final Events': sum(missing_list),
    'Disposition Mismatches': sum(mismatch_list),
    'Total Errors': sum(total_errors_list)
}])
summary_df = pd.concat([summary_df, grand], ignore_index=True)

# Convert numeric columns to int
for c in ['Missing Final Events', 'Disposition Mismatches', 'Total Errors']:
    summary_df[c] = summary_df[c].astype(int)

print('\nSummary table:')
print(summary_df.to_string())

# ---- Write Excel ----
with pd.ExcelWriter('Returns_Disposition_Audit.xlsx', engine='openpyxl') as writer:
    # RawData - exact copy of plan
    plan.to_excel(writer, sheet_name='RawData', index=False)
    
    # Formatted Data - first 8 cols + 4 new cols, exact headers
    fd = formatted[plan_cols + ['Missing Final Event', 'Disposition Mismatch', 'Total Errors', 'Error Summary']].copy()
    # Rename first 8 columns to exact required names
    rename_map = {
        plan_cols[0]: 'Return ID',
        plan_cols[1]: 'Line ID',
        plan_cols[2]: 'Planned Disposition',
        plan_cols[3]: 'Reason Code',
        plan_cols[4]: 'Requested Qty',
        plan_cols[5]: 'Warehouse',
        plan_cols[6]: 'Carrier',
        plan_cols[7]: 'Lane'
    }
    fd = fd.rename(columns=rename_map)
    fd.to_excel(writer, sheet_name='Formatted Data', index=False)
    
    # Summary
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print('\nExcel written successfully.')

# ---- Identify high-priority return IDs ----
error_by_return = formatted.groupby(plan_return_id)['Total Errors'].sum()
top_returns = error_by_return.sort_values(ascending=False).head(5)
print('\nTop error Return IDs:')
print(top_returns)
top_return_ids = list(top_returns.index[:2])

# ---- Write Word document ----
total_missing = sum(missing_list)
total_mismatch = sum(mismatch_list)
total_total = sum(total_errors_list)

doc = Document()
doc.add_heading('Returns Disposition Audit Brief', level=1)

paragraph_text = (
    f'This audit evaluated returns disposition accuracy across two key checks. '
    f'A "Missing Final Event" flags any return line that lacks a completed disposition event in the event log, '
    f'indicating the return was never finalized. '
    f'A "Disposition Mismatch" flags cases where the final completed disposition differs from the originally planned disposition, '
    f'suggesting a processing deviation. '
    f'Across the dataset, we identified {total_missing} Missing Final Events, '
    f'{total_mismatch} Disposition Mismatches, and {total_total} Total Errors. '
    f'Return IDs {top_return_ids[0]} and {top_return_ids[1]} were among the highest-priority cases with frequent exceptions. '
    f'We recommend implementing real-time disposition validation at the warehouse scan point to catch mismatches before shipment, '
    f'and conducting a root-cause review for returns with missing final events to close process gaps.'
)

doc.add_paragraph(paragraph_text)
doc.save('Returns_Disposition_Brief.docx')
print('\nWord document written successfully.')
print('DONE')
```

Run the script:
```bash
cd /root && python3 solve.py
```

## Step 3: Post-execution validation

After the script runs, verify:
1. Check that both output files exist and are non-empty
2. Open the Excel file and verify sheet names are exactly: RawData, Formatted Data, Summary
3. Verify the Formatted Data sheet has exactly 12 columns with the correct headers
4. Verify the Summary sheet has the Grand Total row as the last row
5. Verify the Word document contains numeric totals

```bash
python3 -c "
import openpyxl
from docx import Document
import os

# Check files exist
for f in ['Returns_Disposition_Audit.xlsx', 'Returns_Disposition_Brief.docx']:
    print(f'{f}: exists={os.path.exists(f)}, size={os.path.getsize(f)}')

# Check Excel
wb = openpyxl.load_workbook('Returns_Disposition_Audit.xlsx')
print(f'Sheet names: {wb.sheetnames}')
assert wb.sheetnames == ['RawData', 'Formatted Data', 'Summary'], f'Wrong sheets: {wb.sheetnames}'

# Check Formatted Data headers
ws = wb['Formatted Data']
headers = [ws.cell(1, c).value for c in range(1, ws.max_column+1)]
print(f'Formatted Data headers: {headers}')
expected = ['Return ID', 'Line ID', 'Planned Disposition', 'Reason Code', 'Requested Qty', 'Warehouse', 'Carrier', 'Lane', 'Missing Final Event', 'Disposition Mismatch', 'Total Errors', 'Error Summary']
assert headers == expected, f'Header mismatch: {headers}'
print(f'Formatted Data rows (excl header): {ws.max_row - 1}')

# Check Summary
ws2 = wb['Summary']
sh = [ws2.cell(1, c).value for c in range(1, ws2.max_column+1)]
print(f'Summary headers: {sh}')
last_row = ws2.max_row
print(f'Summary last row: {[ws2.cell(last_row, c).value for c in range(1, ws2.max_column+1)]}')
assert ws2.cell(last_row, 1).value == 'Grand Total'

# Check Word
doc = Document('Returns_Disposition_Brief.docx')
text = ' '.join([p.text for p in doc.paragraphs])
print(f'Word doc length: {len(text)} chars')
print(f'Word doc preview: {text[:300]}')
wb.close()
print('All validations passed.')
"
```

## CRITICAL NOTES from previous failure:

1. **Event filtering**: Make sure to filter for `Event Status == 'COMPLETED'` using case-insensitive comparison. The status column name may vary - find it dynamically.
2. **Latest event**: When multiple COMPLETED events exist for the same (Return ID, Line ID), keep only the LATEST one. Check for date/timestamp columns first; if none exist, use the original row order (last row = latest).
3. **Alias mapping direction**: Carefully check which column in Disposition_Alias.xlsx is the alias and which is the standard name. The alias column maps TO the standard. Apply this mapping to the Final Disposition from the event log before comparing with Planned Disposition.
4. **Column name matching**: The plan file column names may not exactly match the required output headers. Rename them to the exact required names.
5. **Numeric values**: Write concrete integers (0 or 1) for the error flag columns, not formulas.
6. **The Formatted Data must have the SAME number of rows as RawData** (same row order, one row per plan line).

IMPORTANT: During Step 1 inspection, pay very close attention to the alias file structure. Print ALL rows. Determine definitively which column is 'alias' and which is 'standard'. Also inspect the event log carefully - print all unique values of the status column and all column names.

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