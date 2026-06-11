# Task Instruction

Execute the following Python script to produce both deliverables. The script reads the source workbook, applies SLA rules dynamically, builds the three-sheet Excel audit file, and writes the Word executive summary.

```python
import pandas as pd
from openpyxl import Workbook
from docx import Document

# ── 1. Read source data ──────────────────────────────────────────────
tickets = pd.read_excel('/root/Ticket_Queue.xlsx', sheet_name='Tickets')
sla_rules = pd.read_excel('/root/Ticket_Queue.xlsx', sheet_name='SLA_Rules')

# Inspect columns to handle any naming variations
print('Tickets columns:', tickets.columns.tolist())
print('SLA_Rules columns:', sla_rules.columns.tolist())
print('Tickets shape:', tickets.shape)
print('SLA_Rules:')
print(sla_rules.to_string())
print('\nFirst 5 tickets:')
print(tickets.head().to_string())

# ── 2. Build lookup dicts from SLA_Rules ─────────────────────────────
# Map Priority Tier -> Max Open Hours
max_hours = dict(zip(sla_rules['Priority Tier'], sla_rules['Max Open Hours']))
# Map Priority Tier -> Escalation Required (Y/N)
esc_required = dict(zip(sla_rules['Priority Tier'], sla_rules['Escalation Required']))

print('\nMax Hours lookup:', max_hours)
print('Escalation Required lookup:', esc_required)

# ── 3. Build Formatted Data ──────────────────────────────────────────
fd = tickets.copy()

# Keep first 8 columns in specified order
base_cols = ['Ticket ID', 'Queue', 'Priority Tier', 'Open Age Hours',
             'Owner', 'Escalation Code', 'Region', 'Analyst']
fd = fd[base_cols].copy()

# SLA Breach: 1 if Open Age Hours > Max Open Hours for that Priority Tier
fd['SLA Breach'] = fd.apply(
    lambda r: 1 if r['Open Age Hours'] > max_hours.get(r['Priority Tier'], float('inf')) else 0,
    axis=1
)

# Missing Escalation: 1 if Escalation Required == 'Y' for that tier AND Escalation Code is blank
fd['Missing Escalation'] = fd.apply(
    lambda r: 1 if esc_required.get(r['Priority Tier'], 'N') == 'Y'
                   and (pd.isna(r['Escalation Code']) or str(r['Escalation Code']).strip() == '')
              else 0,
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

print('\nFormatted Data value counts for Error Summary:')
print(fd['Error Summary'].value_counts())
print('\nSLA Breaches total:', fd['SLA Breach'].sum())
print('Missing Escalations total:', fd['Missing Escalation'].sum())
print('Total Errors total:', fd['Total Errors'].sum())

# ── 4. Build Summary ─────────────────────────────────────────────────
agg = fd.groupby(['Queue', 'Region'], as_index=False).agg(
    **{'SLA Breaches': ('SLA Breach', 'sum'),
       'Missing Escalations': ('Missing Escalation', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
)
# Keep only groups with Total Errors > 0
agg = agg[agg['Total Errors'] > 0].copy()
# Sort by Queue asc, Region asc
agg = agg.sort_values(['Queue', 'Region']).reset_index(drop=True)

# Grand Total row
grand = pd.DataFrame([{
    'Queue': 'Grand Total',
    'Region': '-',
    'SLA Breaches': fd['SLA Breach'].sum(),
    'Missing Escalations': fd['Missing Escalation'].sum(),
    'Total Errors': fd['Total Errors'].sum()
}])
summary = pd.concat([agg, grand], ignore_index=True)

# Ensure integer types for numeric columns
for c in ['SLA Breaches', 'Missing Escalations', 'Total Errors']:
    summary[c] = summary[c].astype(int)

print('\nSummary table:')
print(summary.to_string())

# ── 5. Write Excel workbook ──────────────────────────────────────────
outpath = '/root/Service_Queue_SLA_Audit.xlsx'
with pd.ExcelWriter(outpath, engine='openpyxl') as writer:
    # RawData – exact copy of Tickets
    tickets.to_excel(writer, sheet_name='RawData', index=False)
    # Formatted Data
    fd.to_excel(writer, sheet_name='Formatted Data', index=False)
    # Summary
    summary.to_excel(writer, sheet_name='Summary', index=False)

print(f'\nExcel written to {outpath}')

# ── 6. Identify top queues for Word brief ────────────────────────────
queue_errors = fd.groupby('Queue')['Total Errors'].sum().sort_values(ascending=False)
top_queues = queue_errors[queue_errors > 0].head(2).index.tolist()
print('Top queues by errors:', top_queues)

total_sla = int(fd['SLA Breach'].sum())
total_missing = int(fd['Missing Escalation'].sum())
total_errors = int(fd['Total Errors'].sum())

# ── 7. Write Word brief ──────────────────────────────────────────────
doc = Document()
doc.add_heading('Service Queue SLA Audit – Executive Summary', level=1)

para = (
    f'This audit evaluated ticket queue health using two automated checks: '
    f'an SLA Breach flag, which triggers when a ticket\'s Open Age Hours exceeds the '
    f'maximum threshold defined for its Priority Tier, and a Missing Escalation flag, '
    f'which triggers when a ticket\'s Priority Tier requires escalation but no Escalation Code is recorded. '
    f'Across the dataset, the audit identified {total_sla} SLA Breaches, '
    f'{total_missing} Missing Escalations, and {total_errors} Total Errors. '
    f'The queues with the most frequent exceptions were {", ".join(top_queues)}, '
    f'which should be prioritized for staffing reviews and process improvements. '
    f'We recommend implementing automated escalation routing for high-priority tiers '
    f'and establishing real-time SLA dashboards to reduce breach rates.'
)
doc.add_paragraph(para)

docpath = '/root/Service_Queue_SLA_Brief.docx'
doc.save(docpath)
print(f'Word brief written to {docpath}')

# ── 8. Verification ──────────────────────────────────────────────────
# Re-read and verify
import openpyxl
wb = openpyxl.load_workbook(outpath, read_only=True)
print('\nSheet names in output:', wb.sheetnames)
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'  {sn}: {ws.max_row} rows x {ws.max_column} cols')
    # Print header row
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    print(f'    Headers: {headers}')
wb.close()

from docx import Document as DocReader
doc2 = DocReader(docpath)
for p in doc2.paragraphs:
    if p.text.strip():
        print(f'Word para: {p.text[:200]}...')
print('\nDone.')
```

After running the script, verify:
1. `/root/Service_Queue_SLA_Audit.xlsx` exists with exactly three sheets: `RawData`, `Formatted Data`, `Summary`.
2. `RawData` has the same row count as the source Tickets sheet.
3. `Formatted Data` has 12 columns with exact headers as specified, and all computed columns contain concrete values (not formulas).
4. `Summary` is filtered to groups with Total Errors > 0, sorted by Queue then Region, and ends with a Grand Total row.
5. `/root/Service_Queue_SLA_Brief.docx` contains definitions of both checks, the three numeric totals, at least two queue names, and a recommendation.

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