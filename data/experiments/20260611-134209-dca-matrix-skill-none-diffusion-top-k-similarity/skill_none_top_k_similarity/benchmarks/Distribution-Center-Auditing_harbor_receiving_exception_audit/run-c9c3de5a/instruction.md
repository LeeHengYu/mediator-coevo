# Task Instruction

Execute the following Python script to produce both deliverables. The script reads the source workbook, builds the three-sheet Excel audit file, and writes the Word executive summary.

```python
import openpyxl
from openpyxl import Workbook
from docx import Document

# ── 1. Read source workbook ──────────────────────────────────────────────
src = openpyxl.load_workbook('/root/Receiving_Log.xlsx', data_only=True)
src_ws = src.active
rows_all = list(src_ws.iter_rows(values_only=True))
header_raw = list(rows_all[0])
data_rows = rows_all[1:]

# Case-insensitive column lookup
def col_idx(name):
    target = name.strip().lower()
    for i, h in enumerate(header_raw):
        if str(h).strip().lower() == target:
            return i
    raise KeyError(f'Column {name!r} not found in {header_raw}')

IDX_RECEIPT   = col_idx('Receipt ID')
IDX_ITEM      = col_idx('Item Code')
IDX_EXPECTED  = col_idx('Expected Qty')
IDX_RECEIVED  = col_idx('Received Qty')
IDX_STORAGE   = col_idx('Storage Class')
IDX_TEMP      = col_idx('Temp Status')
IDX_SUPPLIER  = col_idx('Supplier')
IDX_DOCK      = col_idx('Dock')

ORDERED_COLS = [IDX_RECEIPT, IDX_ITEM, IDX_EXPECTED, IDX_RECEIVED,
                IDX_STORAGE, IDX_TEMP, IDX_SUPPLIER, IDX_DOCK]

# ── 2. Compute derived columns ───────────────────────────────────────────
formatted_header = ['Receipt ID','Item Code','Expected Qty','Received Qty',
                    'Storage Class','Temp Status','Supplier','Dock',
                    'Qty Variance','Cold Chain Error','Total Errors','Error Summary']

formatted_rows = []
for row in data_rows:
    base = [row[c] for c in ORDERED_COLS]
    expected = row[IDX_EXPECTED]
    received = row[IDX_RECEIVED]
    storage  = str(row[IDX_STORAGE]).strip().upper()
    temp     = str(row[IDX_TEMP]).strip().upper()

    qty_var = 1 if received != expected else 0
    cold_err = 1 if storage in ('CHILLED', 'FROZEN') and temp != 'OK' else 0
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

# ── 3. Build Summary aggregation ─────────────────────────────────────────
from collections import defaultdict
agg = defaultdict(lambda: [0, 0, 0])  # qty_var, cold_err, total
for fr in formatted_rows:
    item_code = fr[1]
    supplier  = fr[6]
    key = (item_code, supplier)
    agg[key][0] += fr[8]
    agg[key][1] += fr[9]
    agg[key][2] += fr[10]

summary_rows = []
for (ic, sup), vals in agg.items():
    if vals[2] > 0:
        summary_rows.append([ic, sup] + vals)

summary_rows.sort(key=lambda r: (str(r[0]), str(r[1])))

total_qty = sum(r[2] for r in summary_rows)
total_cold = sum(r[3] for r in summary_rows)
total_all = sum(r[4] for r in summary_rows)
summary_rows.append(['Grand Total', '-', total_qty, total_cold, total_all])

summary_header = ['Item Code','Supplier','Qty Variance Errors',
                  'Cold Chain Errors','Total Errors']

# ── 4. Write Excel workbook ──────────────────────────────────────────────
wb = Workbook()

# RawData sheet
ws_raw = wb.active
ws_raw.title = 'RawData'
ws_raw.append(header_raw)
for row in data_rows:
    ws_raw.append(list(row))

# Formatted Data sheet
ws_fmt = wb.create_sheet('Formatted Data')
ws_fmt.append(formatted_header)
for row in formatted_rows:
    ws_fmt.append(row)

# Summary sheet
ws_sum = wb.create_sheet('Summary')
ws_sum.append(summary_header)
for row in summary_rows:
    ws_sum.append(row)

wb.save('/root/Receiving_Exception_Audit.xlsx')
print('Excel saved.')

# ── 5. Identify top item codes for the brief ─────────────────────────────
item_totals = defaultdict(int)
for fr in formatted_rows:
    item_totals[fr[1]] += fr[10]
top_items = sorted(item_totals.items(), key=lambda x: -x[1])
top_codes = [ic for ic, cnt in top_items if cnt > 0][:2]

# ── 6. Write Word brief ──────────────────────────────────────────────────
doc = Document()
doc.add_heading('Receiving Exception Brief', level=1)

para = (
    f'This audit reviewed every inbound receiving record for two exception types. '
    f'A Qty Variance error flags any line where the Received Qty differs from the Expected Qty. '
    f'A Cold Chain Error flags any CHILLED or FROZEN item whose Temp Status is not OK. '
    f'Across all records the audit identified {total_qty} Qty Variance errors, '
    f'{total_cold} Cold Chain errors, and {total_all} Total Errors. '
    f'Item codes {top_codes[0]} and {top_codes[1]} had the most frequent exceptions and should be prioritized for root-cause investigation. '
    f'We recommend tightening dock-level temperature verification for cold-chain items and reconciling purchase-order quantities with suppliers before dispatch to reduce recurring variances.'
)
doc.add_paragraph(para)
doc.save('/root/Receiving_Exception_Brief.docx')
print('Word brief saved.')
print('Done.')
```

After running the script, verify:
1. `/root/Receiving_Exception_Audit.xlsx` exists and contains exactly three sheets named `RawData`, `Formatted Data`, `Summary`.
2. The `Formatted Data` sheet has 12 columns with the exact headers specified.
3. The `Summary` sheet's last row has `Item Code` = `Grand Total` and `Supplier` = `-`.
4. `/root/Receiving_Exception_Brief.docx` exists and contains the phrases 'Qty Variance' and 'Cold Chain Error' along with numeric totals and a recommendation.

If the script fails due to missing packages, install them first with `pip install openpyxl python-docx`.

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