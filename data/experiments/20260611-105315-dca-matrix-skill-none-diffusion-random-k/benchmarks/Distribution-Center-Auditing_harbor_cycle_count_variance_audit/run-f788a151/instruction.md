# Task Instruction

Execute the following Python script to produce both deliverables. Before writing the main script, first inspect the input files to understand their structure.

```bash
cd /root
python3 << 'INSPECT'
import pandas as pd

# Inspect Cycle_Plan.xlsx
xl1 = pd.ExcelFile('Cycle_Plan.xlsx')
print('=== Cycle_Plan sheets:', xl1.sheet_names)
for s in xl1.sheet_names:
    df = pd.read_excel(xl1, s)
    print(f'--- Sheet: {s}, shape: {df.shape}')
    print(df.columns.tolist())
    print(df.head(5))
    print()

# Inspect Count_Event_Log.xlsx
xl2 = pd.ExcelFile('Count_Event_Log.xlsx')
print('=== Count_Event_Log sheets:', xl2.sheet_names)
for s in xl2.sheet_names:
    df = pd.read_excel(xl2, s)
    print(f'--- Sheet: {s}, shape: {df.shape}')
    print(df.columns.tolist())
    print(df.head(10))
    print()

# Inspect Cycle_Template.xlsx
xl3 = pd.ExcelFile('Cycle_Template.xlsx')
print('=== Cycle_Template sheets:', xl3.sheet_names)
for s in xl3.sheet_names:
    df = pd.read_excel(xl3, s)
    print(f'--- Sheet: {s}, shape: {df.shape}')
    print(df.columns.tolist())
    print(df.head(5))
    print()
INSPECT
```

After inspecting the output, run the main processing script. Adapt column names based on what you see in the inspection output. The script should follow this logic:

```python
import pandas as pd
import shutil
from docx import Document

# Step 1: Copy template to output to preserve Overview sheet
shutil.copy('Cycle_Template.xlsx', 'Cycle_Count_Variance_Audit.xlsx')

# Step 2: Read input data
plan_df = pd.read_excel('Cycle_Plan.xlsx')  # adjust sheet name if needed
event_df = pd.read_excel('Count_Event_Log.xlsx')  # adjust sheet name if needed

# Step 3: Prepare RawData - exact copy of plan table
raw_data = plan_df.copy()

# Step 4: Prepare Formatted Data
# Normalize column names for matching (map actual names to expected)
# The first 7 columns should be: Facility, Session ID, Bin ID, Product ID, Expected Qty, Allowed Variance, Approval Needed
# Rename if needed based on inspection

formatted = plan_df.copy()
# Ensure first 7 columns are named exactly as required
# formatted.columns = ['Facility', 'Session ID', 'Bin ID', 'Product ID', 'Expected Qty', 'Allowed Variance', 'Approval Needed'] + list of any extras

# Process event log: filter to Event Type == 'FINAL', drop rows with blank keys or blank Count Qty
# Keep only latest row per (Facility, Session ID, Bin ID) - use last occurrence or sort by timestamp if available
final_events = event_df[event_df['Event Type'].str.strip().str.upper() == 'FINAL'].copy()
# Drop rows where Facility, Session ID, Bin ID, or Count Qty is blank/NaN
final_events = final_events.dropna(subset=['Facility', 'Session ID', 'Bin ID', 'Count Qty'])
# For blank strings too:
for col in ['Facility', 'Session ID', 'Bin ID']:
    final_events = final_events[final_events[col].astype(str).str.strip() != '']
# Keep latest per key (last row in original order, or sort by timestamp if exists)
final_events = final_events.drop_duplicates(subset=['Facility', 'Session ID', 'Bin ID'], keep='last')

# Create lookup dict: (Facility, Session ID, Bin ID) -> Count Qty
final_lookup = {}
for _, row in final_events.iterrows():
    key = (str(row['Facility']).strip(), str(row['Session ID']).strip(), str(row['Bin ID']).strip())
    final_lookup[key] = row['Count Qty']

# Compute new columns
missing_final = []
approval_gap = []
total_errors = []
error_summary = []

for _, row in formatted.iterrows():
    key = (str(row['Facility']).strip(), str(row['Session ID']).strip(), str(row['Bin ID']).strip())
    has_final = key in final_lookup
    
    mfc = 0 if has_final else 1
    
    ag = 0
    if has_final and str(row['Approval Needed']).strip().upper() == 'YES':
        count_qty = final_lookup[key]
        expected = row['Expected Qty']
        allowed = row['Allowed Variance']
        if abs(count_qty - expected) > allowed:
            ag = 1
    
    te = mfc + ag
    
    if mfc == 1 and ag == 1:
        es = 'Missing Final Count, Approval Gap'
    elif mfc == 1:
        es = 'Missing Final Count'
    elif ag == 1:
        es = 'Approval Gap'
    else:
        es = 'None'
    
    missing_final.append(mfc)
    approval_gap.append(ag)
    total_errors.append(te)
    error_summary.append(es)

formatted['Missing Final Count'] = missing_final
formatted['Approval Gap'] = approval_gap
formatted['Total Errors'] = total_errors
formatted['Error Summary'] = error_summary

# Step 5: Summary sheet
summary = formatted.groupby(['Facility', 'Session ID']).agg(
    **{'Missing Final Counts': ('Missing Final Count', 'sum'),
       'Approval Gaps': ('Approval Gap', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
).reset_index()
summary = summary[summary['Total Errors'] > 0]
summary = summary.sort_values(['Facility', 'Session ID']).reset_index(drop=True)

# Grand Total row
grand = pd.DataFrame([{
    'Facility': 'Grand Total',
    'Session ID': '-',
    'Missing Final Counts': summary['Missing Final Counts'].sum(),
    'Approval Gaps': summary['Approval Gaps'].sum(),
    'Total Errors': summary['Total Errors'].sum()
}])
summary = pd.concat([summary, grand], ignore_index=True)

# Ensure numeric columns are int
for c in ['Missing Final Counts', 'Approval Gaps', 'Total Errors']:
    summary[c] = summary[c].astype(int)

# Step 6: Write to Excel using overlay mode to preserve Overview
with pd.ExcelWriter('Cycle_Count_Variance_Audit.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    raw_data.to_excel(writer, sheet_name='RawData', index=False)
    formatted.to_excel(writer, sheet_name='Formatted Data', index=False)
    summary.to_excel(writer, sheet_name='Summary', index=False)

# Step 7: Create Word document
total_mfc = int(grand['Missing Final Counts'].iloc[0])
total_ag = int(grand['Approval Gaps'].iloc[0])
total_te = int(grand['Total Errors'].iloc[0])

# Find top 2 facility-session combos by Total Errors
top = summary[summary['Facility'] != 'Grand Total'].nlargest(2, 'Total Errors')
high_priority_pairs = [f"{r['Facility']}-{r['Session ID']} ({int(r['Total Errors'])} errors)" for _, r in top.iterrows()]

doc = Document()
doc.add_heading('Cycle Count Variance Audit – Executive Brief', level=1)

para = (
    f"This audit assessed inventory accuracy through two checks: "
    f"'Missing Final Count' flags bins where no final count event was recorded, "
    f"and 'Approval Gap' flags bins where the variance between expected and counted quantities "
    f"exceeded the allowed threshold despite requiring approval. "
    f"Across all facilities, the audit identified {total_mfc} Missing Final Counts, "
    f"{total_ag} Approval Gaps, and {total_te} Total Errors. "
    f"High-priority facility-session combinations include {' and '.join(high_priority_pairs)}, "
    f"which should be investigated first. "
    f"We recommend implementing automated count-completion alerts and tightening approval workflows "
    f"for high-variance bins to reduce recurrence of these exceptions."
)
doc.add_paragraph(para)
doc.save('Cycle_Count_Variance_Brief.docx')

print('Done. Files created.')
```

IMPORTANT ADAPTATION NOTES:
- After the inspection step, adapt column names in the main script to match the actual column names found in the input files. The column names above (e.g., 'Event Type', 'Count Qty', 'Facility', 'Session ID', 'Bin ID') are guesses — use the real names.
- If the event log has a timestamp column, sort by it before `drop_duplicates(keep='last')` to ensure 'latest' is correct.
- Ensure the first 7 columns of 'Formatted Data' are named exactly: Facility, Session ID, Bin ID, Product ID, Expected Qty, Allowed Variance, Approval Needed. Rename them if the source uses different names.
- After writing, verify the output by reading back the Excel file and checking sheet names, column names, row counts, and the Summary grand total row. Also verify the Word file exists and has content.

Verification step:
```python
import pandas as pd
from docx import Document

xl = pd.ExcelFile('Cycle_Count_Variance_Audit.xlsx')
print('Sheets:', xl.sheet_names)
for s in xl.sheet_names:
    df = pd.read_excel(xl, s)
    print(f'\n--- {s}: {df.shape}')
    print(df.columns.tolist())
    if s == 'Summary':
        print(df)
    elif s == 'Formatted Data':
        print(df[['Missing Final Count','Approval Gap','Total Errors','Error Summary']].value_counts())
        print(df.head())

doc = Document('Cycle_Count_Variance_Brief.docx')
for p in doc.paragraphs:
    print(p.text)
```

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