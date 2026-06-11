# Task Instruction

Execute the following Python script in a single step to produce both deliverables.

```python
import shutil
import pandas as pd
from openpyxl import load_workbook
from docx import Document

# Step 1: Read source data
manifest = pd.read_excel('/root/Manifest_Plan.xlsx')
dock_scan = pd.read_excel('/root/Dock_Scan_Log.xlsx')

# Step 2: Copy template to preserve Overview sheet
shutil.copy('/root/Outbound_Audit_Template.xlsx', '/root/Outbound_Load_Audit.xlsx')

# Step 3: Process dock scan log - keep only latest LOADED scan per (Shipment ID, Carton ID)
loaded_scans = dock_scan[dock_scan['Status'] == 'LOADED'].copy()
# Keep the last occurrence (latest) for each (Shipment ID, Carton ID)
loaded_scans = loaded_scans.drop_duplicates(subset=['Shipment ID', 'Carton ID'], keep='last')

# Step 4: Build Formatted Data
# Map manifest columns to the required 8-column schema
formatted = manifest.copy()
# Ensure column names match the required schema exactly
required_cols = ['Shipment ID', 'Carton ID', 'Planned Zone', 'Route', 'Expected Weight', 'Hazmat Flag', 'Carrier', 'Wave']
print('Manifest columns:', list(manifest.columns))
print('Dock scan columns:', list(dock_scan.columns))

# Rename columns if needed to match required schema
col_mapping = {}
for rc in required_cols:
    if rc not in formatted.columns:
        # Try case-insensitive match
        for mc in formatted.columns:
            if mc.lower().replace('_', ' ') == rc.lower():
                col_mapping[mc] = rc
                break
if col_mapping:
    formatted = formatted.rename(columns=col_mapping)
    print('Applied column mapping:', col_mapping)

# Select only the first 8 columns in the required order
formatted = formatted[required_cols].copy()

# Merge with loaded scans to check for missing scans and zone mismatches
merged = formatted.merge(
    loaded_scans[['Shipment ID', 'Carton ID', 'Scanned Zone']],
    on=['Shipment ID', 'Carton ID'],
    how='left'
)

# Compute error columns
merged['Missing Load Scan'] = merged['Scanned Zone'].isna().astype(int)
merged['Zone Mismatch'] = ((merged['Scanned Zone'].notna()) & (merged['Scanned Zone'] != merged['Planned Zone'])).astype(int)
merged['Total Errors'] = merged['Missing Load Scan'] + merged['Zone Mismatch']

def error_summary(row):
    parts = []
    if row['Missing Load Scan'] == 1:
        parts.append('Missing Load Scan')
    if row['Zone Mismatch'] == 1:
        parts.append('Zone Mismatch')
    return ', '.join(parts) if parts else 'None'

merged['Error Summary'] = merged.apply(error_summary, axis=1)

# Drop the helper Scanned Zone column
formatted_data = merged.drop(columns=['Scanned Zone'])

print('Formatted Data shape:', formatted_data.shape)
print('Formatted Data columns:', list(formatted_data.columns))
print('Error counts:', formatted_data[['Missing Load Scan', 'Zone Mismatch', 'Total Errors']].sum().to_dict())

# Step 5: Build Summary
summary_agg = formatted_data.groupby(['Route', 'Shipment ID']).agg(
    **{'Missing Load Scans': ('Missing Load Scan', 'sum'),
       'Zone Mismatches': ('Zone Mismatch', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
).reset_index()

# Filter to only groups with Total Errors > 0
summary_agg = summary_agg[summary_agg['Total Errors'] > 0].copy()

# Sort by Route ascending, then Shipment ID ascending
summary_agg = summary_agg.sort_values(['Route', 'Shipment ID']).reset_index(drop=True)

# Compute grand totals from the full Formatted Data (not just filtered summary)
total_missing = int(formatted_data['Missing Load Scan'].sum())
total_zone = int(formatted_data['Zone Mismatch'].sum())
total_errors = int(formatted_data['Total Errors'].sum())

# Append Grand Total row
grand_total_row = pd.DataFrame([{
    'Route': 'Grand Total',
    'Shipment ID': '-',
    'Missing Load Scans': total_missing,
    'Zone Mismatches': total_zone,
    'Total Errors': total_errors
}])
summary_final = pd.concat([summary_agg, grand_total_row], ignore_index=True)

print('Summary shape:', summary_final.shape)
print(summary_final.to_string())

# Step 6: Write Excel workbook (overlay mode to preserve Overview)
with pd.ExcelWriter('/root/Outbound_Load_Audit.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    manifest.to_excel(writer, sheet_name='RawData', index=False)
    formatted_data.to_excel(writer, sheet_name='Formatted Data', index=False)
    summary_final.to_excel(writer, sheet_name='Summary', index=False)

# Verify the workbook has the correct sheets
wb = load_workbook('/root/Outbound_Load_Audit.xlsx')
print('Final workbook sheets:', wb.sheetnames)
wb.close()

# Step 7: Find top shipment IDs with most errors for the brief
shipment_errors = formatted_data.groupby('Shipment ID')['Total Errors'].sum().sort_values(ascending=False)
top_shipments = shipment_errors[shipment_errors > 0].head(2)
top_ids = list(top_shipments.index)
print('Top error shipments:', top_ids, top_shipments.to_dict())

# Step 8: Create Word document
doc = Document()
doc.add_heading('Outbound Load Audit Brief', level=1)

paragraph_text = (
    f'This audit evaluated outbound carton handoff accuracy across all planned shipments. '
    f'A "Missing Load Scan" error indicates that a carton listed in the manifest plan was never recorded '
    f'with a LOADED status in the dock scan log, meaning it may not have been physically loaded onto the truck. '
    f'A "Zone Mismatch" error indicates that a carton was scanned as loaded but in a different dock zone than '
    f'the one originally planned, suggesting a potential mis-sort or routing error. '
    f'Across the dataset, the audit identified {total_missing} Missing Load Scan(s), '
    f'{total_zone} Zone Mismatch(es), and {total_errors} Total Error(s). '
    f'Shipment IDs {top_ids[0]} and {top_ids[1] if len(top_ids) > 1 else top_ids[0]} had the highest frequency of exceptions and should be prioritized for investigation. '
    f'It is recommended to implement real-time scan validation alerts at dock zones to catch missing and misrouted cartons '
    f'before trailer doors are closed, reducing handoff discrepancies in future waves.'
)

doc.add_paragraph(paragraph_text)
doc.save('/root/Outbound_Load_Brief.docx')
print('Word document saved successfully.')
print('DONE')
```

After running the script, verify:
1. `/root/Outbound_Load_Audit.xlsx` exists and contains sheets: Overview, RawData, Formatted Data, Summary.
2. `/root/Outbound_Load_Brief.docx` exists.
3. The printed output shows correct error counts and sheet names.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=hard, tags=[excel, openpyxl, docx, audit, logistics].
Verifier config: timeout_sec=900.0.