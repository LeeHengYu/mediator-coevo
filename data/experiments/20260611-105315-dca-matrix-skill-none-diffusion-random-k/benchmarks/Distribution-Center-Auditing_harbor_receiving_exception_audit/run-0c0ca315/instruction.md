# Task Instruction

Execute the following Python script to produce both deliverables. Before running, inspect the source workbook to confirm column names, then run the script and validate outputs.

```bash
cd /root
python3 << 'PYEOF'
import pandas as pd
from docx import Document

# ── 1. Read source ──────────────────────────────────────────────────────────
df = pd.read_excel('/root/Receiving_Log.xlsx')
print('Source columns:', list(df.columns))
print('Source shape:', df.shape)
print('First 3 rows:')
print(df.head(3))

# ── 2. Standardize column mapping ──────────────────────────────────────────
# Expected canonical headers
canonical = ['Receipt ID','Item Code','Expected Qty','Received Qty',
             'Storage Class','Temp Status','Supplier','Dock']

# Build mapping: try exact match first, then case-insensitive / stripped
src_cols = list(df.columns)
mapping = {}
for canon in canonical:
    for sc in src_cols:
        if sc.strip().lower() == canon.strip().lower():
            mapping[sc] = canon
            break

df_raw = df.rename(columns=mapping)
# Ensure all canonical columns present
for c in canonical:
    assert c in df_raw.columns, f'Missing column: {c}'

# ── 3. RawData sheet ───────────────────────────────────────────────────────
raw_data = df_raw.copy()

# ── 4. Formatted Data sheet ────────────────────────────────────────────────
fd = df_raw[canonical].copy()

# Qty Variance: 1 if Received != Expected, else 0
fd['Qty Variance'] = (fd['Received Qty'] != fd['Expected Qty']).astype(int)

# Cold Chain Error: 1 when Storage Class in {CHILLED, FROZEN} (case-insensitive)
# AND Temp Status is NOT 'OK' (case-insensitive)
is_cold = fd['Storage Class'].astype(str).str.strip().str.upper().isin(['CHILLED','FROZEN'])
temp_ok = fd['Temp Status'].astype(str).str.strip().str.upper() == 'OK'
fd['Cold Chain Error'] = ((is_cold) & (~temp_ok)).astype(int)

# Total Errors
fd['Total Errors'] = fd['Qty Variance'] + fd['Cold Chain Error']

# Error Summary
def error_summary(row):
    parts = []
    if row['Qty Variance'] == 1:
        parts.append('Qty Variance')
    if row['Cold Chain Error'] == 1:
        parts.append('Cold Chain Error')
    return ', '.join(parts) if parts else 'None'

fd['Error Summary'] = fd.apply(error_summary, axis=1)

print('\nFormatted Data columns:', list(fd.columns))
print('Formatted Data shape:', fd.shape)
print('Qty Variance total:', fd['Qty Variance'].sum())
print('Cold Chain Error total:', fd['Cold Chain Error'].sum())
print('Total Errors total:', fd['Total Errors'].sum())

# ── 5. Summary sheet ───────────────────────────────────────────────────────
agg = fd.groupby(['Item Code','Supplier'], as_index=False).agg(
    **{'Qty Variance Errors': ('Qty Variance','sum'),
       'Cold Chain Errors': ('Cold Chain Error','sum'),
       'Total Errors': ('Total Errors','sum')}
)

# Keep only groups with Total Errors > 0
agg = agg[agg['Total Errors'] > 0].copy()

# Sort by Item Code asc, Supplier asc
agg = agg.sort_values(['Item Code','Supplier']).reset_index(drop=True)

# Grand Total row
grand = pd.DataFrame([{
    'Item Code': 'Grand Total',
    'Supplier': '-',
    'Qty Variance Errors': agg['Qty Variance Errors'].sum(),
    'Cold Chain Errors': agg['Cold Chain Errors'].sum(),
    'Total Errors': agg['Total Errors'].sum()
}])
summary = pd.concat([agg, grand], ignore_index=True)

# Ensure integer types for numeric columns
for col in ['Qty Variance Errors','Cold Chain Errors','Total Errors']:
    summary[col] = summary[col].astype(int)

print('\nSummary shape:', summary.shape)
print(summary.to_string(index=False))

# ── 6. Write Excel ─────────────────────────────────────────────────────────
with pd.ExcelWriter('/root/Receiving_Exception_Audit.xlsx', engine='openpyxl') as w:
    raw_data.to_excel(w, sheet_name='RawData', index=False)
    fd.to_excel(w, sheet_name='Formatted Data', index=False)
    summary.to_excel(w, sheet_name='Summary', index=False)

print('\nExcel written. Verifying sheets...')
import openpyxl
wb = openpyxl.load_workbook('/root/Receiving_Exception_Audit.xlsx')
print('Sheet names:', wb.sheetnames)
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'  {sn}: {ws.max_row} rows x {ws.max_column} cols, headers={[c.value for c in ws[1]]}')
wb.close()

# ── 7. Word Brief ──────────────────────────────────────────────────────────
total_qty_var = int(fd['Qty Variance'].sum())
total_cc_err = int(fd['Cold Chain Error'].sum())
total_errors = int(fd['Total Errors'].sum())

# Find top item codes by total errors
top_items = fd.groupby('Item Code')['Total Errors'].sum().sort_values(ascending=False)
top2 = list(top_items[top_items > 0].head(2).index)
top2_str = ' and '.join(top2) if len(top2) >= 2 else (top2[0] if top2 else 'N/A')

doc = Document()
doc.add_heading('Receiving Exception Brief', level=1)

para = (
    f'This audit reviewed all inbound receiving records for quantity and cold-chain compliance. '
    f'A Qty Variance error is flagged whenever the Received Qty differs from the Expected Qty, '
    f'indicating a potential shortage or overage. '
    f'A Cold Chain Error is flagged when a CHILLED or FROZEN item arrives with a Temp Status other than OK, '
    f'signaling a possible temperature-control breach during transit. '
    f'Across the dataset, {total_qty_var} Qty Variance errors, {total_cc_err} Cold Chain errors, '
    f'and {total_errors} Total Errors were identified. '
    f'Item codes {top2_str} showed the highest frequency of exceptions and should be prioritized '
    f'for supplier performance reviews. '
    f'We recommend implementing pre-shipment quantity verification and installing continuous '
    f'temperature loggers on all cold-chain loads to reduce future discrepancies.'
)
doc.add_paragraph(para)
doc.save('/root/Receiving_Exception_Brief.docx')
print('\nWord document written.')
print('DONE')
PYEOF
```

After the script completes, verify both output files exist:
```bash
ls -la /root/Receiving_Exception_Audit.xlsx /root/Receiving_Exception_Brief.docx
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