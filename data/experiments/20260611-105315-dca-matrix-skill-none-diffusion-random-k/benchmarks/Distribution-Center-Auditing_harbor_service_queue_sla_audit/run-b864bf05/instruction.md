# Task Instruction

Execute the following steps in a single Python script to produce the two deliverables.

## Step 0 – Inspect the source workbook
```python
import openpyxl
wb = openpyxl.load_workbook('/root/Ticket_Queue.xlsx')
for s in wb.sheetnames:
    ws = wb[s]
    print(f'--- {s} ---')
    for row in ws.iter_rows(min_row=1, max_row=min(5, ws.max_row), values_only=False):
        print([c.value for c in row])
    print(f'max_row={ws.max_row}, max_col={ws.max_column}')
```
Run this first, read the output carefully, then proceed with the processing script below (adjust column references if the inspection reveals different names).

## Step 1 – Full processing script

```python
import openpyxl
from openpyxl.utils import get_column_letter
from copy import copy
import pandas as pd
from docx import Document

# ── 1. Read source data ──────────────────────────────────────────────
src = '/root/Ticket_Queue.xlsx'

# Read Tickets sheet
df_tickets = pd.read_excel(src, sheet_name='Tickets')

# Read SLA_Rules sheet
df_sla = pd.read_excel(src, sheet_name='SLA_Rules')

# Print columns for verification
print('Tickets columns:', list(df_tickets.columns))
print('SLA_Rules columns:', list(df_sla.columns))
print('SLA_Rules:')
print(df_sla.to_string())

# ── 2. Build lookup from SLA_Rules ───────────────────────────────────
# Build dict: Priority Tier -> {Max Open Hours, Escalation Required}
sla_lookup = {}
for _, r in df_sla.iterrows():
    tier = r['Priority Tier']
    sla_lookup[tier] = {
        'Max Open Hours': r['Max Open Hours'],
        'Escalation Required': str(r['Escalation Required']).strip().upper()
    }
print('SLA lookup:', sla_lookup)

# ── 3. Build Formatted Data ──────────────────────────────────────────
# Keep first 8 columns in required order
required_cols = ['Ticket ID', 'Queue', 'Priority Tier', 'Open Age Hours',
                 'Owner', 'Escalation Code', 'Region', 'Analyst']
df_fmt = df_tickets[required_cols].copy()

# Compute flags
def calc_sla_breach(row):
    tier = row['Priority Tier']
    if tier in sla_lookup:
        return 1 if row['Open Age Hours'] > sla_lookup[tier]['Max Open Hours'] else 0
    return 0

def calc_missing_esc(row):
    tier = row['Priority Tier']
    if tier in sla_lookup:
        if sla_lookup[tier]['Escalation Required'] == 'Y':
            val = row['Escalation Code']
            if pd.isna(val) or str(val).strip() == '':
                return 1
    return 0

df_fmt['SLA Breach'] = df_fmt.apply(calc_sla_breach, axis=1)
df_fmt['Missing Escalation'] = df_fmt.apply(calc_missing_esc, axis=1)
df_fmt['Total Errors'] = df_fmt['SLA Breach'] + df_fmt['Missing Escalation']

def error_summary(row):
    b = row['SLA Breach']
    m = row['Missing Escalation']
    if b == 1 and m == 1:
        return 'SLA Breach, Missing Escalation'
    elif b == 1:
        return 'SLA Breach'
    elif m == 1:
        return 'Missing Escalation'
    else:
        return 'None'

df_fmt['Error Summary'] = df_fmt.apply(error_summary, axis=1)

print('Formatted Data sample:')
print(df_fmt.head(10).to_string())
print(f'Total SLA Breaches: {df_fmt["SLA Breach"].sum()}')
print(f'Total Missing Escalations: {df_fmt["Missing Escalation"].sum()}')
print(f'Total Errors: {df_fmt["Total Errors"].sum()}')

# ── 4. Build Summary ─────────────────────────────────────────────────
grp = df_fmt.groupby(['Queue', 'Region'], as_index=False).agg(
    **{'SLA Breaches': ('SLA Breach', 'sum'),
       'Missing Escalations': ('Missing Escalation', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
)
grp = grp[grp['Total Errors'] > 0].copy()
grp = grp.sort_values(['Queue', 'Region']).reset_index(drop=True)

# Cast to int
for c in ['SLA Breaches', 'Missing Escalations', 'Total Errors']:
    grp[c] = grp[c].astype(int)

# Grand Total row
grand = pd.DataFrame([{
    'Queue': 'Grand Total',
    'Region': '-',
    'SLA Breaches': int(df_fmt['SLA Breach'].sum()),
    'Missing Escalations': int(df_fmt['Missing Escalation'].sum()),
    'Total Errors': int(df_fmt['Total Errors'].sum())
}])
df_summary = pd.concat([grp, grand], ignore_index=True)

print('Summary:')
print(df_summary.to_string())

# ── 5. Write Excel workbook ──────────────────────────────────────────
out_xlsx = '/root/Service_Queue_SLA_Audit.xlsx'
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
    df_tickets.to_excel(writer, sheet_name='RawData', index=False)
    df_fmt.to_excel(writer, sheet_name='Formatted Data', index=False)
    df_summary.to_excel(writer, sheet_name='Summary', index=False)

print(f'Written {out_xlsx}')

# ── 6. Verify Excel ──────────────────────────────────────────────────
wb_check = openpyxl.load_workbook(out_xlsx)
print('Sheets:', wb_check.sheetnames)
for s in wb_check.sheetnames:
    ws = wb_check[s]
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column+1)]
    print(f'{s} headers: {headers}, rows={ws.max_row}')
    if s == 'Summary':
        last_row = [ws.cell(ws.max_row, c).value for c in range(1, ws.max_column+1)]
        print(f'  Last row: {last_row}')

# ── 7. Identify top queues for Word doc ──────────────────────────────
top_queues = grp.sort_values('Total Errors', ascending=False).head(5)
print('Top queues by errors:')
print(top_queues.to_string())

# Pick at least two high-priority queue names
top_queue_names = top_queues['Queue'].unique()[:3]

total_sla = int(df_fmt['SLA Breach'].sum())
total_miss = int(df_fmt['Missing Escalation'].sum())
total_err = int(df_fmt['Total Errors'].sum())

# ── 8. Write Word document ───────────────────────────────────────────
out_docx = '/root/Service_Queue_SLA_Brief.docx'
doc = Document()
doc.add_heading('Service Queue SLA Audit – Executive Summary', level=1)

q_list = ', '.join(top_queue_names[:2]) if len(top_queue_names) >= 2 else str(top_queue_names[0])
if len(top_queue_names) >= 3:
    q_extra = top_queue_names[2]
else:
    q_extra = None

para_text = (
    f'This audit reviewed open service-desk tickets against the defined SLA thresholds. '
    f'An SLA Breach is flagged when a ticket\'s Open Age Hours exceeds the maximum allowed '
    f'hours for its Priority Tier as defined in the SLA rules. '
    f'A Missing Escalation is flagged when the SLA rules require escalation for a given '
    f'Priority Tier but the ticket\'s Escalation Code is blank. '
    f'Across all tickets, the audit identified {total_sla} SLA Breaches, '
    f'{total_miss} Missing Escalations, and {total_err} Total Errors. '
    f'The queues with the most frequent exceptions include {q_list}'
)
if q_extra:
    para_text += f' and {q_extra}'
para_text += (
    f', which should be prioritized for corrective action. '
    f'We recommend implementing automated escalation triggers and periodic SLA threshold '
    f'reviews to reduce recurring breaches and improve queue health.'
)

doc.add_paragraph(para_text)
doc.save(out_docx)
print(f'Written {out_docx}')
print('DONE')
```

Run the inspection step first. If the column names in the source workbook differ from those assumed above (e.g., 'Priority Tier', 'Open Age Hours', 'Escalation Code', 'Max Open Hours', 'Escalation Required'), update the column references in the processing script accordingly before running it. After the processing script completes, verify:
1. The output Excel has exactly three sheets: RawData, Formatted Data, Summary.
2. Formatted Data has 12 columns with the exact headers specified.
3. Summary last row is 'Grand Total' / '-' / totals.
4. The Word document exists and contains the required content.

If any verification fails, diagnose and fix before finishing.

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