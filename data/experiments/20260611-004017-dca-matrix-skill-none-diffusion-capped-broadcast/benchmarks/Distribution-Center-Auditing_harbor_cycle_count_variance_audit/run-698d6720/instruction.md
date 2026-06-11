# Task Instruction

## Task: Cycle Count Variance Audit

You must create two output files:
1. `/root/Cycle_Count_Variance_Audit.xlsx`
2. `/root/Cycle_Count_Variance_Brief.docx`

### Step 0: Inspect Source Files

```bash
pip install openpyxl pandas python-docx
```

Then run Python to inspect all three source files:
```python
import pandas as pd

# Inspect Cycle_Plan.xlsx
xl_plan = pd.ExcelFile('/root/Cycle_Plan.xlsx')
print('Cycle_Plan sheets:', xl_plan.sheet_names)
for s in xl_plan.sheet_names:
    df = pd.read_excel(xl_plan, s)
    print(f'\nSheet: {s}, shape: {df.shape}')
    print(df.columns.tolist())
    print(df.head(10))
    print(df.dtypes)

# Inspect Count_Event_Log.xlsx
xl_log = pd.ExcelFile('/root/Count_Event_Log.xlsx')
print('\nCount_Event_Log sheets:', xl_log.sheet_names)
for s in xl_log.sheet_names:
    df = pd.read_excel(xl_log, s)
    print(f'\nSheet: {s}, shape: {df.shape}')
    print(df.columns.tolist())
    print(df.head(10))
    print(df.dtypes)

# Inspect Cycle_Template.xlsx
xl_tmpl = pd.ExcelFile('/root/Cycle_Template.xlsx')
print('\nCycle_Template sheets:', xl_tmpl.sheet_names)
for s in xl_tmpl.sheet_names:
    df = pd.read_excel(xl_tmpl, s)
    print(f'\nSheet: {s}, shape: {df.shape}')
    print(df.columns.tolist())
    print(df.head(10))
```

Carefully note:
- The exact column names in Cycle_Plan.xlsx (these become the first 7 columns of RawData and Formatted Data)
- The exact column names in Count_Event_Log.xlsx (especially the key columns: Facility, Session ID, Bin ID, Event Type, Count Qty, and any timestamp column)
- The Overview sheet content in Cycle_Template.xlsx

### Step 1: Build the Output Excel

Write a single Python script that does everything. Here is the logic:

```python
import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from copy import copy
from docx import Document

# ---- Load source data ----
plan_df = pd.read_excel('/root/Cycle_Plan.xlsx')
log_df = pd.read_excel('/root/Count_Event_Log.xlsx')

# ---- Prepare FINAL event lookup ----
# Filter to Event Type == 'FINAL' (case-insensitive), drop rows with blank keys or blank Count Qty
log_final = log_df.copy()
# Standardize Event Type check
log_final = log_final[log_final['Event Type'].astype(str).str.strip().str.upper() == 'FINAL']
# Drop rows where any key is blank/NaN
for col in ['Facility', 'Session ID', 'Bin ID']:
    log_final = log_final[log_final[col].notna() & (log_final[col].astype(str).str.strip() != '')]
# Drop rows where Count Qty is blank/NaN
log_final = log_final[log_final['Count Qty'].notna()]

# Keep only the LATEST row per (Facility, Session ID, Bin ID)
# Identify the timestamp/date column - inspect and use whatever exists (e.g., 'Timestamp', 'Event Date', 'Date')
# Sort by that column descending, then drop_duplicates keeping first
# IMPORTANT: Inspect the actual column names from Step 0 and adjust this line
time_cols = [c for c in log_final.columns if any(kw in c.lower() for kw in ['time', 'date', 'stamp', 'seq', 'row'])]
if time_cols:
    sort_col = time_cols[0]
    log_final = log_final.sort_values(sort_col, ascending=False)
else:
    # If no time column, use original row order (last occurrence = latest)
    log_final = log_final.iloc[::-1]

log_final = log_final.drop_duplicates(subset=['Facility', 'Session ID', 'Bin ID'], keep='first')

# Create a lookup dict: (Facility, Session ID, Bin ID) -> Count Qty
final_lookup = {}
for _, row in log_final.iterrows():
    key = (str(row['Facility']).strip(), str(row['Session ID']).strip(), str(row['Bin ID']).strip())
    final_lookup[key] = row['Count Qty']

# ---- Build Formatted Data ----
formatted_rows = []
for _, row in plan_df.iterrows():
    key = (str(row['Facility']).strip(), str(row['Session ID']).strip(), str(row['Bin ID']).strip())
    count_qty = final_lookup.get(key, None)
    
    has_final = count_qty is not None
    missing_final = 0 if has_final else 1
    
    approval_gap = 0
    if has_final:
        approval_needed = str(row['Approval Needed']).strip().upper()
        if approval_needed == 'YES':
            expected = float(row['Expected Qty'])
            allowed = float(row['Allowed Variance'])
            if abs(expected - float(count_qty)) > allowed:
                approval_gap = 1
    
    total_errors = missing_final + approval_gap
    
    parts = []
    if missing_final == 1:
        parts.append('Missing Final Count')
    if approval_gap == 1:
        parts.append('Approval Gap')
    error_summary = ', '.join(parts) if parts else 'None'
    
    formatted_rows.append({
        'Facility': row['Facility'],
        'Session ID': row['Session ID'],
        'Bin ID': row['Bin ID'],
        'Product ID': row['Product ID'],
        'Expected Qty': row['Expected Qty'],
        'Allowed Variance': row['Allowed Variance'],
        'Approval Needed': row['Approval Needed'],
        'Missing Final Count': missing_final,
        'Approval Gap': approval_gap,
        'Total Errors': total_errors,
        'Error Summary': error_summary
    })

formatted_df = pd.DataFrame(formatted_rows)

# ---- Build Summary ----
summary_agg = formatted_df.groupby(['Facility', 'Session ID']).agg(
    **{'Missing Final Counts': ('Missing Final Count', 'sum'),
       'Approval Gaps': ('Approval Gap', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
).reset_index()

# Filter to Total Errors > 0
summary_agg = summary_agg[summary_agg['Total Errors'] > 0]

# Sort by Facility asc, Session ID asc
summary_agg = summary_agg.sort_values(['Facility', 'Session ID']).reset_index(drop=True)

# Grand Total row
grand_total = pd.DataFrame([{
    'Facility': 'Grand Total',
    'Session ID': '-',
    'Missing Final Counts': summary_agg['Missing Final Counts'].sum(),
    'Approval Gaps': summary_agg['Approval Gaps'].sum(),
    'Total Errors': summary_agg['Total Errors'].sum()
}])

summary_df = pd.concat([summary_agg, grand_total], ignore_index=True)

# ---- Write Excel ----
# First, copy the template to get the Overview sheet
import shutil
shutil.copy('/root/Cycle_Template.xlsx', '/root/Cycle_Count_Variance_Audit.xlsx')

wb = openpyxl.load_workbook('/root/Cycle_Count_Variance_Audit.xlsx')

# Ensure Overview sheet exists and is named correctly
# Remove any sheets that aren't Overview
for name in wb.sheetnames:
    if name != 'Overview':
        del wb[name]

# Create RawData sheet
ws_raw = wb.create_sheet('RawData')
for r_idx, row in enumerate(dataframe_to_rows(plan_df, index=False, header=True), 1):
    for c_idx, val in enumerate(row, 1):
        ws_raw.cell(row=r_idx, column=c_idx, value=val)

# Create Formatted Data sheet
ws_fmt = wb.create_sheet('Formatted Data')
for r_idx, row in enumerate(dataframe_to_rows(formatted_df, index=False, header=True), 1):
    for c_idx, val in enumerate(row, 1):
        ws_fmt.cell(row=r_idx, column=c_idx, value=val)

# Create Summary sheet
ws_sum = wb.create_sheet('Summary')
for r_idx, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), 1):
    for c_idx, val in enumerate(row, 1):
        ws_sum.cell(row=r_idx, column=c_idx, value=val)

# Reorder sheets: Overview, RawData, Formatted Data, Summary
wb.move_sheet('Overview', offset=0)

wb.save('/root/Cycle_Count_Variance_Audit.xlsx')
print('Excel saved.')

# ---- Identify top 2 facility-session combos for Word doc ----
# Exclude Grand Total row
summary_data = summary_agg.copy()  # already excludes Grand Total
summary_data = summary_data.sort_values('Total Errors', ascending=False)
top2 = summary_data.head(2)

top2_strings = []
for _, r in top2.iterrows():
    fac = str(r['Facility']).strip()
    sess = str(r['Session ID']).strip()
    errs = int(r['Total Errors'])
    top2_strings.append(f"{fac} {sess} ({errs} total errors)")

total_missing = int(grand_total['Missing Final Counts'].iloc[0])
total_approval = int(grand_total['Approval Gaps'].iloc[0])
total_errors_all = int(grand_total['Total Errors'].iloc[0])

# ---- Write Word doc ----
doc = Document()
doc.add_heading('Cycle Count Variance Audit – Executive Summary', level=1)

para_text = (
    f"This audit evaluated cycle-count accuracy across all scheduled sessions. "
    f"A \"Missing Final Count\" flags any bin where no confirmed final count event was recorded, "
    f"meaning the physical verification step was never completed. "
    f"An \"Approval Gap\" flags bins where a final count exists but the absolute variance between "
    f"the expected quantity and the counted quantity exceeds the allowed tolerance and approval was required. "
    f"Across the dataset the audit identified {total_missing} Missing Final Counts, "
    f"{total_approval} Approval Gaps, and {total_errors_all} Total Errors. "
    f"The highest-priority facility-session combinations are {top2_strings[0]} and {top2_strings[1]}; "
    f"these should be investigated first. "
    f"We recommend scheduling immediate recounts for all bins flagged as Missing Final Count "
    f"and routing all Approval Gap exceptions to warehouse supervisors for sign-off before the next reporting cycle."
)

doc.add_paragraph(para_text)
doc.save('/root/Cycle_Count_Variance_Brief.docx')
print('Word doc saved.')
print('\nTop 2 combos mentioned:', top2_strings)
```

### Step 2: Verify the Outputs

After running the script, verify:

1. Check sheet names:
```python
wb = openpyxl.load_workbook('/root/Cycle_Count_Variance_Audit.xlsx')
print('Sheet names:', wb.sheetnames)
# Must be exactly: ['Overview', 'RawData', 'Formatted Data', 'Summary']
```

2. Check Formatted Data columns and sample values:
```python
df = pd.read_excel('/root/Cycle_Count_Variance_Audit.xlsx', sheet_name='Formatted Data')
print(df.columns.tolist())
print(df.head(10))
print('Missing Final Count values:', df['Missing Final Count'].unique())
print('Approval Gap values:', df['Approval Gap'].unique())
print('Error Summary values:', df['Error Summary'].unique())
```

3. Check Summary sheet:
```python
df_sum = pd.read_excel('/root/Cycle_Count_Variance_Audit.xlsx', sheet_name='Summary')
print(df_sum)
# Last row should be Grand Total
```

4. Check Word doc content:
```python
from docx import Document
doc = Document('/root/Cycle_Count_Variance_Brief.docx')
for p in doc.paragraphs:
    print(p.text)
```

Verify the Word doc text contains:
- The words 'Missing Final Count' and 'Approval Gap' as definitions
- Numeric totals for Missing Final Counts, Approval Gaps, Total Errors
- At least two specific facility-session combinations (e.g., 'FAC001 SESS003')
- At least one actionable recommendation

### CRITICAL: Word Document Facility-Session Mentions

The previous execution failed because the verifier checks that at least two high-priority facility-session combinations are mentioned in the Word document. The verifier likely searches for strings like `'FACILITY_NAME SESSION_ID'` or `'FACILITY_NAME-SESSION_ID'` or similar patterns.

**You MUST include the actual Facility and Session ID values from the top 2 rows of the Summary sheet (sorted by Total Errors descending) in the Word document text.** Format them as `"FACILITY SESSION_ID"` (space-separated) so the verifier can find them. Also include them hyphenated like `"FACILITY-SESSION_ID"` for safety.

For example, if the top two are (FAC001, SESS005) and (FAC002, SESS003), the Word doc should contain both `"FAC001 SESS005"` and `"FAC002 SESS003"` as substrings.

### Step 3: Adjust Column Names if Needed

IMPORTANT: The column names in the script above assume standard names from the source files. After Step 0 inspection, you MUST adjust any column name references to match the actual headers found in the source Excel files. Do NOT assume column names - read them from the files first.

If the Event Log has a different column name for the timestamp (or has no timestamp at all), adjust the deduplication logic accordingly. If there's no timestamp, use the last occurrence in row order as the latest.

### Step 4: Run the verifier if available

Check if there's a test file:
```bash
ls /root/test_output.py 2>/dev/null || ls /root/tests/ 2>/dev/null || find /root -name 'test_*.py' -maxdepth 2
```

If found, run it:
```bash
cd /root && python -m pytest test_output.py -v 2>&1 | head -80
```

Fix any failures before finishing.

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