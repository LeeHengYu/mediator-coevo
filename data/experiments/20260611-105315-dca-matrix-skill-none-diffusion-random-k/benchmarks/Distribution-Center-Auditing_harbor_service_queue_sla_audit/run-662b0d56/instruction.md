# Task Instruction

Execute the following Python script to produce both deliverables. Before writing the script, first inspect the source workbook to understand its exact structure.

## Step 1: Inspect the source data
```bash
cd /root
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('Ticket_Queue.xlsx')
print('Sheet names:', wb.sheetnames)
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'\n--- {sn} ---')
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        print(row)
        if i >= 5:
            print('...')
            break
    print(f'Total rows: {ws.max_row}, Total cols: {ws.max_column}')
"
```

## Step 2: After inspecting, run the main generation script
```python
import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from docx import Document

# ── Load source data ──
tickets = pd.read_excel('/root/Ticket_Queue.xlsx', sheet_name='Tickets')
sla_rules = pd.read_excel('/root/Ticket_Queue.xlsx', sheet_name='SLA_Rules')

print('Tickets columns:', list(tickets.columns))
print('Tickets shape:', tickets.shape)
print('SLA_Rules columns:', list(sla_rules.columns))
print(sla_rules)
print('Tickets head:')
print(tickets.head())

# ── Build SLA lookup dictionaries ──
# Keyed by Priority Tier
max_hours = dict(zip(sla_rules['Priority Tier'], sla_rules['Max Open Hours']))
esc_required = dict(zip(sla_rules['Priority Tier'], sla_rules['Escalation Required']))

print('max_hours:', max_hours)
print('esc_required:', esc_required)

# ── Build Formatted Data ──
# Keep first 8 columns in specified order
first_8_cols = ['Ticket ID', 'Queue', 'Priority Tier', 'Open Age Hours',
                'Owner', 'Escalation Code', 'Region', 'Analyst']

fd = tickets[first_8_cols].copy()

# SLA Breach: 1 if Open Age Hours > Max Open Hours for that Priority Tier
fd['SLA Breach'] = fd.apply(
    lambda r: 1 if r['Open Age Hours'] > max_hours.get(r['Priority Tier'], float('inf')) else 0,
    axis=1
)

# Missing Escalation: 1 if Escalation Required is Y for that tier AND Escalation Code is blank/NaN
fd['Missing Escalation'] = fd.apply(
    lambda r: 1 if esc_required.get(r['Priority Tier'], 'N') == 'Y' and (
        pd.isna(r['Escalation Code']) or str(r['Escalation Code']).strip() == ''
    ) else 0,
    axis=1
)

# Total Errors
fd['Total Errors'] = fd['SLA Breach'] + fd['Missing Escalation']

# Error Summary
def error_summary(row):
    parts = []
    if row['SLA Breach'] == 1:
        parts.append('SLA Breach')
    if row['Missing Escalation'] == 1:
        parts.append('Missing Escalation')
    return ', '.join(parts) if parts else 'None'

fd['Error Summary'] = fd.apply(error_summary, axis=1)

print('\nFormatted Data columns:', list(fd.columns))
print(fd.head(10))

# ── Build Summary ──
# Aggregate by (Queue, Region) from Formatted Data
agg = fd.groupby(['Queue', 'Region'], sort=False).agg(
    **{'SLA Breaches': ('SLA Breach', 'sum'),
       'Missing Escalations': ('Missing Escalation', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
).reset_index()

# Filter only groups with Total Errors > 0
agg = agg[agg['Total Errors'] > 0].copy()

# Sort by Queue asc, then Region asc
agg = agg.sort_values(['Queue', 'Region']).reset_index(drop=True)

# Grand Total row from full dataset
grand_total = pd.DataFrame([{
    'Queue': 'Grand Total',
    'Region': '-',
    'SLA Breaches': fd['SLA Breach'].sum(),
    'Missing Escalations': fd['Missing Escalation'].sum(),
    'Total Errors': fd['Total Errors'].sum()
}])

summary = pd.concat([agg, grand_total], ignore_index=True)

# Ensure integer types
for col in ['SLA Breaches', 'Missing Escalations', 'Total Errors']:
    summary[col] = summary[col].astype(int)

print('\nSummary:')
print(summary)

# ── Write Excel workbook ──
with pd.ExcelWriter('/root/Service_Queue_SLA_Audit.xlsx', engine='openpyxl') as writer:
    # RawData: exact copy of Tickets
    tickets.to_excel(writer, sheet_name='RawData', index=False)
    # Formatted Data
    fd.to_excel(writer, sheet_name='Formatted Data', index=False)
    # Summary
    summary.to_excel(writer, sheet_name='Summary', index=False)

print('\nExcel written successfully.')

# ── Verify Excel ──
wb = openpyxl.load_workbook('/root/Service_Queue_SLA_Audit.xlsx')
print('Output sheets:', wb.sheetnames)
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'{sn}: {ws.max_row} rows x {ws.max_column} cols')
    # Print header
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column+1)]
    print(f'  Headers: {headers}')

# ── Identify top queues with errors for the Word doc ──
queue_errors = fd.groupby('Queue')['Total Errors'].sum().sort_values(ascending=False)
top_queues = queue_errors[queue_errors > 0].head(5)
print('\nTop queues by errors:')
print(top_queues)

total_sla = int(fd['SLA Breach'].sum())
total_missing = int(fd['Missing Escalation'].sum())
total_errors = int(fd['Total Errors'].sum())

# Pick at least two high-priority queues
top_queue_names = list(top_queues.index[:2])

# ── Write Word document ──
doc = Document()
doc.add_heading('Service Queue SLA Audit – Executive Brief', level=1)

para_text = (
    f'This audit reviewed the service-desk ticket queue against SLA thresholds defined per priority tier. '
    f'An "SLA Breach" is flagged when a ticket\'s Open Age Hours exceeds the maximum allowed hours for its Priority Tier as defined in the SLA rules. '
    f'A "Missing Escalation" is flagged when the SLA rules require escalation for a given Priority Tier (Escalation Required = Y) but the ticket\'s Escalation Code is blank. '
    f'Across the dataset, the audit identified {total_sla} SLA Breaches, {total_missing} Missing Escalations, and {total_errors} Total Errors. '
    f'The queues with the most frequent exceptions are {", ".join(top_queue_names)}, which should be prioritized for corrective action. '
    f'We recommend implementing automated escalation routing for high-priority tiers and establishing real-time SLA monitoring dashboards to reduce breach rates and ensure timely ticket resolution.'
)

doc.add_paragraph(para_text)
doc.save('/root/Service_Queue_SLA_Brief.docx')
print('\nWord document written successfully.')
print('Done.')
```

## Step 3: Verify both output files exist and are valid
```bash
ls -la /root/Service_Queue_SLA_Audit.xlsx /root/Service_Queue_SLA_Brief.docx
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('/root/Service_Queue_SLA_Audit.xlsx')
print('Sheets:', wb.sheetnames)
assert wb.sheetnames == ['RawData', 'Formatted Data', 'Summary'], f'Wrong sheets: {wb.sheetnames}'
for sn in wb.sheetnames:
    ws = wb[sn]
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column+1)]
    print(f'{sn} headers: {headers}')
    print(f'{sn} rows: {ws.max_row}')

# Verify Formatted Data headers
ws = wb['Formatted Data']
expected = ['Ticket ID','Queue','Priority Tier','Open Age Hours','Owner','Escalation Code','Region','Analyst','SLA Breach','Missing Escalation','Total Errors','Error Summary']
actual = [ws.cell(1, c).value for c in range(1, 13)]
assert actual == expected, f'Formatted Data headers mismatch: {actual}'

# Verify Summary headers
ws = wb['Summary']
expected_s = ['Queue','Region','SLA Breaches','Missing Escalations','Total Errors']
actual_s = [ws.cell(1, c).value for c in range(1, 6)]
assert actual_s == expected_s, f'Summary headers mismatch: {actual_s}'

# Verify last row is Grand Total
last_row = ws.max_row
assert ws.cell(last_row, 1).value == 'Grand Total', f'Last row Queue is not Grand Total: {ws.cell(last_row, 1).value}'
assert ws.cell(last_row, 2).value == '-', f'Last row Region is not -: {ws.cell(last_row, 2).value}'
print('All Excel checks passed.')

from docx import Document
doc = Document('/root/Service_Queue_SLA_Brief.docx')
text = ' '.join([p.text for p in doc.paragraphs])
print('Word text length:', len(text))
assert 'SLA Breach' in text
assert 'Missing Escalation' in text
print('Word checks passed.')
print('ALL VERIFICATIONS PASSED')
"
```

IMPORTANT NOTES:
- Run Step 1 first to inspect the actual column names in the source workbook. If column names differ from what the script assumes (e.g., 'Priority Tier' vs 'Priority_Tier'), adjust the script accordingly before running Step 2.
- Ensure `Escalation Code` blank detection handles both NaN and empty strings.
- The Grand Total row sums come from the FULL Formatted Data, not just the filtered summary rows.
- All numeric columns in the added columns (9-12) must be concrete values, not Excel formulas.
- If the top queues list has fewer than 2 entries, adjust the Word document text to mention whatever queues are available.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=hard, tags=[excel, openpyxl, docx, audit, service].
Verifier config: timeout_sec=900.0.