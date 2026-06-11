# Task Instruction

Execute the following Python script to produce both deliverables. Before running, inspect the source workbook to confirm column names, then generate the outputs.

```bash
cd /root && python3 << 'PYEOF'
import openpyxl
from openpyxl import Workbook
from docx import Document
from collections import defaultdict

# ── 1. Read source workbook ──────────────────────────────────────────────
src = openpyxl.load_workbook('Receiving_Log.xlsx', data_only=True)
src_ws = src.active  # assume single sheet

rows = list(src_ws.iter_rows(values_only=True))
headers = list(rows[0])
data = rows[1:]

print('Source columns:', headers)
print('Row count (excl header):', len(data))

# Map column indices (case-insensitive match for robustness)
def col_idx(name):
    name_l = name.strip().lower()
    for i, h in enumerate(headers):
        if str(h).strip().lower() == name_l:
            return i
    raise ValueError(f'Column {name!r} not found in {headers}')

i_receipt   = col_idx('Receipt ID')
i_item      = col_idx('Item Code')
i_exp_qty   = col_idx('Expected Qty')
i_rec_qty   = col_idx('Received Qty')
i_storage   = col_idx('Storage Class')
i_temp      = col_idx('Temp Status')
i_supplier  = col_idx('Supplier')
i_dock      = col_idx('Dock')

# Ordered first-8 column indices
first8_idx = [i_receipt, i_item, i_exp_qty, i_rec_qty, i_storage, i_temp, i_supplier, i_dock]
first8_headers = ['Receipt ID', 'Item Code', 'Expected Qty', 'Received Qty',
                  'Storage Class', 'Temp Status', 'Supplier', 'Dock']

# ── 2. Compute derived columns ───────────────────────────────────────────
formatted_rows = []
for row in data:
    base = [row[j] for j in first8_idx]

    exp_qty = row[i_exp_qty]
    rec_qty = row[i_rec_qty]
    storage = str(row[i_storage]).strip().upper() if row[i_storage] is not None else ''
    temp_st = str(row[i_temp]).strip().upper() if row[i_temp] is not None else ''

    qty_var = 1 if rec_qty != exp_qty else 0
    cold_err = 1 if storage in ('CHILLED', 'FROZEN') and temp_st != 'OK' else 0
    total_err = qty_var + cold_err

    if qty_var and cold_err:
        summary = 'Qty Variance, Cold Chain Error'
    elif qty_var:
        summary = 'Qty Variance'
    elif cold_err:
        summary = 'Cold Chain Error'
    else:
        summary = 'None'

    formatted_rows.append(base + [qty_var, cold_err, total_err, summary])

formatted_headers = first8_headers + ['Qty Variance', 'Cold Chain Error', 'Total Errors', 'Error Summary']

# ── 3. Build Summary aggregation ─────────────────────────────────────────
agg = defaultdict(lambda: [0, 0, 0])  # (item, supplier) -> [qty_var, cold_err, total]
for fr in formatted_rows:
    item_code = fr[1]
    supplier  = fr[6]
    qty_v     = fr[8]
    cold_e    = fr[9]
    tot_e     = fr[10]
    key = (item_code, supplier)
    agg[key][0] += qty_v
    agg[key][1] += cold_e
    agg[key][2] += tot_e

# Filter to groups with total errors > 0
summary_rows = []
for (item, supp), (qv, ce, te) in agg.items():
    if te > 0:
        summary_rows.append([item, supp, qv, ce, te])

# Sort by Item Code asc, Supplier asc
summary_rows.sort(key=lambda r: (str(r[0]), str(r[1])))

# Grand total
grand_qv = sum(r[2] for r in summary_rows)
grand_ce = sum(r[3] for r in summary_rows)
grand_te = sum(r[4] for r in summary_rows)
summary_rows.append(['Grand Total', '-', grand_qv, grand_ce, grand_te])

summary_headers = ['Item Code', 'Supplier', 'Qty Variance Errors', 'Cold Chain Errors', 'Total Errors']

print(f'Total Qty Variance Errors: {grand_qv}')
print(f'Total Cold Chain Errors:   {grand_ce}')
print(f'Total Errors:              {grand_te}')

# ── 4. Write Receiving_Exception_Audit.xlsx ──────────────────────────────
wb = Workbook()

# Sheet 1: RawData
ws_raw = wb.active
ws_raw.title = 'RawData'
ws_raw.append(list(headers))  # original headers exactly
for row in data:
    ws_raw.append(list(row))

# Sheet 2: Formatted Data
ws_fmt = wb.create_sheet('Formatted Data')
ws_fmt.append(formatted_headers)
for fr in formatted_rows:
    ws_fmt.append(fr)

# Sheet 3: Summary
ws_sum = wb.create_sheet('Summary')
ws_sum.append(summary_headers)
for sr in summary_rows:
    ws_sum.append(sr)

wb.save('Receiving_Exception_Audit.xlsx')
print('Saved Receiving_Exception_Audit.xlsx')

# ── 5. Identify top item codes for the brief ─────────────────────────────
item_totals = defaultdict(int)
for sr in summary_rows[:-1]:  # exclude grand total
    item_totals[sr[0]] += sr[4]
top_items = sorted(item_totals.items(), key=lambda x: -x[1])[:2]
top_item_names = [t[0] for t in top_items]
print('Top items:', top_items)

# ── 6. Write Receiving_Exception_Brief.docx ──────────────────────────────
doc = Document()
doc.add_heading('Receiving Exception Brief', level=1)

para = doc.add_paragraph()
para.add_run(
    f'This audit reviewed inbound grocery receiving logs for quantity and cold-chain compliance. '
    f'A Qty Variance error is flagged whenever the Received Qty differs from the Expected Qty, '
    f'indicating a count discrepancy at the dock. '
    f'A Cold Chain Error is flagged when a CHILLED or FROZEN item arrives with a Temp Status other than OK, '
    f'signaling a potential temperature-control breach during transit. '
    f'Across the dataset the audit identified {grand_qv} Qty Variance errors, '
    f'{grand_ce} Cold Chain errors, and {grand_te} Total Errors. '
    f'High-priority item codes with frequent exceptions include {top_item_names[0]} and {top_item_names[1]}, '
    f'which together account for a significant share of all flagged issues. '
    f'We recommend implementing mandatory recount protocols for suppliers with repeated quantity variances '
    f'and installing real-time temperature loggers on all chilled and frozen shipments to reduce cold-chain failures.'
)

doc.save('Receiving_Exception_Brief.docx')
print('Saved Receiving_Exception_Brief.docx')
PYEOF
```

After the script completes, verify:
1. `ls -la /root/Receiving_Exception_Audit.xlsx /root/Receiving_Exception_Brief.docx` — both files exist.
2. Run a quick validation:
```bash
python3 << 'CHECK'
import openpyxl
wb = openpyxl.load_workbook('/root/Receiving_Exception_Audit.xlsx')
print('Sheet names:', wb.sheetnames)
assert wb.sheetnames == ['RawData', 'Formatted Data', 'Summary'], f'Bad sheets: {wb.sheetnames}'

ws_fmt = wb['Formatted Data']
hdrs = [c.value for c in ws_fmt[1]]
print('Formatted Data headers:', hdrs)
expected = ['Receipt ID','Item Code','Expected Qty','Received Qty','Storage Class','Temp Status','Supplier','Dock','Qty Variance','Cold Chain Error','Total Errors','Error Summary']
assert hdrs == expected, f'Header mismatch: {hdrs}'

ws_sum = wb['Summary']
sum_hdrs = [c.value for c in ws_sum[1]]
print('Summary headers:', sum_hdrs)
expected_sum = ['Item Code','Supplier','Qty Variance Errors','Cold Chain Errors','Total Errors']
assert sum_hdrs == expected_sum, f'Summary header mismatch: {sum_hdrs}'

# Check last row is Grand Total
last_row = [c.value for c in ws_sum[ws_sum.max_row]]
print('Last summary row:', last_row)
assert last_row[0] == 'Grand Total' and last_row[1] == '-', f'Grand Total row wrong: {last_row}'

print('All checks passed.')
CHECK
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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=medium, tags=[excel, openpyxl, docx, audit, warehouse].
Verifier config: timeout_sec=900.0.