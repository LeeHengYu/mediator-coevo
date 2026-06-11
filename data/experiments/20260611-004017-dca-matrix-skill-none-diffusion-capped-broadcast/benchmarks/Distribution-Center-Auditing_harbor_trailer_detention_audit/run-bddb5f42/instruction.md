# Task Instruction

Execute the following Python script to produce both deliverables. Before writing the script, inspect the source workbook to discover exact sheet names and column headers.

## Step 0 – Inspect the source file
```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('/root/Trailer_Detention_Log.xlsx')
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'Sheet: {sn}, rows={ws.max_row}, cols={ws.max_column}')
    for row in ws.iter_rows(min_row=1, max_row=min(3, ws.max_row), values_only=False):
        print([c.value for c in row])
"
```
Record the sheet name and the exact header strings. Then map them to the 8 required output columns:
1. Load ID
2. Carrier
3. Allowed Hold Hours
4. Actual Hold Hours
5. Seal Required
6. Seal Status
7. Yard
8. Dispatcher

## Step 1 – Build the full generation script
Write and run `/root/build.py` with the following logic:

```python
import pandas as pd
from openpyxl import load_workbook, Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from docx import Document

# --- Read source ---
src = '/root/Trailer_Detention_Log.xlsx'
df_raw = pd.read_excel(src)  # adjust sheet_name if needed after Step 0

# --- RawData: exact copy ---
# We will write this as-is.

# --- Formatted Data ---
# Map source columns to required names (adjust mapping based on Step 0 inspection)
# Example mapping (update keys to match actual source headers):
rename_map = {}  # fill after inspection, e.g. {'Load_ID': 'Load ID', ...}
# If source headers already match, rename_map can be empty.

df = df_raw.copy()
if rename_map:
    df.rename(columns=rename_map, inplace=True)

# Ensure required columns exist
required = ['Load ID','Carrier','Allowed Hold Hours','Actual Hold Hours',
            'Seal Required','Seal Status','Yard','Dispatcher']
for c in required:
    assert c in df.columns, f'Missing column: {c}'

# Compute new columns
df['Detention Overrun'] = (df['Actual Hold Hours'] > df['Allowed Hold Hours']).astype(int)
df['Seal Error'] = df.apply(
    lambda r: 1 if str(r['Seal Required']).strip().upper() == 'YES'
                   and str(r['Seal Status']).strip().upper() != 'VERIFIED'
              else 0, axis=1)
df['Total Errors'] = df['Detention Overrun'] + df['Seal Error']

def error_summary(row):
    parts = []
    if row['Detention Overrun'] == 1:
        parts.append('Detention Overrun')
    if row['Seal Error'] == 1:
        parts.append('Seal Error')
    return ', '.join(parts) if parts else 'None'

df['Error Summary'] = df.apply(error_summary, axis=1)

# Keep only the 12 required columns in order
formatted_cols = required + ['Detention Overrun','Seal Error','Total Errors','Error Summary']
df_fmt = df[formatted_cols]

# --- Summary ---
agg = df_fmt.groupby(['Carrier','Yard'], sort=False).agg(
    **{'Detention Overrun Errors': ('Detention Overrun','sum'),
       'Seal Errors': ('Seal Error','sum'),
       'Total Errors': ('Total Errors','sum')}
).reset_index()
agg = agg[agg['Total Errors'] > 0]
agg = agg.sort_values(['Carrier','Yard'], ascending=[True,True]).reset_index(drop=True)

grand = pd.DataFrame([{
    'Carrier': 'Grand Total',
    'Yard': '-',
    'Detention Overrun Errors': agg['Detention Overrun Errors'].sum(),
    'Seal Errors': agg['Seal Errors'].sum(),
    'Total Errors': agg['Total Errors'].sum()
}])
df_summary = pd.concat([agg, grand], ignore_index=True)

# --- Write Excel ---
out_xlsx = '/root/Trailer_Detention_Audit.xlsx'
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
    df_raw.to_excel(writer, sheet_name='RawData', index=False)
    df_fmt.to_excel(writer, sheet_name='Formatted Data', index=False)
    df_summary.to_excel(writer, sheet_name='Summary', index=False)

print('Excel written.')

# --- Identify top carriers for the brief ---
carrier_totals = df_fmt.groupby('Carrier')['Total Errors'].sum().sort_values(ascending=False)
top_carriers = carrier_totals[carrier_totals > 0].head(2).index.tolist()

total_det = int(df_fmt['Detention Overrun'].sum())
total_seal = int(df_fmt['Seal Error'].sum())
total_err = int(df_fmt['Total Errors'].sum())

# --- Write Word ---
doc = Document()
doc.add_heading('Trailer Detention Audit – Executive Summary', level=1)

p1 = (f'This audit reviewed trailer detention compliance across all yards. '
      f'A Detention Overrun is flagged when a trailer\'s Actual Hold Hours exceed its Allowed Hold Hours. '
      f'A Seal Error is flagged when a trailer requires a seal (Seal Required = YES) but the Seal Status is not VERIFIED.')

p2 = (f'Across the dataset, there were {total_det} Detention Overrun error(s), '
      f'{total_seal} Seal Error(s), and {total_err} Total Error(s).')

if len(top_carriers) >= 2:
    p3 = (f'The carriers with the most frequent exceptions are {top_carriers[0]} and {top_carriers[1]}. '
          f'We recommend prioritizing corrective action plans for these carriers, '
          f'including stricter hold-time enforcement and mandatory seal verification checks before trailer release.')
else:
    p3 = (f'The carrier with the most frequent exceptions is {top_carriers[0] if top_carriers else "N/A"}. '
          f'We recommend implementing stricter hold-time enforcement and mandatory seal verification checks before trailer release.')

doc.add_paragraph(p1)
doc.add_paragraph(p2)
doc.add_paragraph(p3)
doc.save('/root/Trailer_Detention_Brief.docx')
print('Word written.')
```

## Step 2 – Validate outputs
After running the script, verify:
1. `/root/Trailer_Detention_Audit.xlsx` exists and has exactly three sheets named `RawData`, `Formatted Data`, `Summary`.
2. `Formatted Data` has 12 columns with the exact headers specified.
3. `Summary` ends with a `Grand Total` row.
4. `/root/Trailer_Detention_Brief.docx` exists and contains the required content.

Run:
```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('/root/Trailer_Detention_Audit.xlsx')
print('Sheets:', wb.sheetnames)
for sn in wb.sheetnames:
    ws = wb[sn]
    headers = [c.value for c in ws[1]]
    print(f'{sn} headers: {headers}')
    print(f'{sn} rows: {ws.max_row}')
    if sn == 'Summary':
        last = [c.value for c in ws[ws.max_row]]
        print(f'Last row: {last}')
from docx import Document
doc = Document('/root/Trailer_Detention_Brief.docx')
for p in doc.paragraphs:
    print(p.text[:120])
"
```

If the source headers don't match the required output headers exactly, update the `rename_map` dictionary in the script accordingly and re-run. Ensure all values in the computed columns are concrete (int/str), not formulas.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=medium, tags=[excel, openpyxl, docx, audit, logistics].
Verifier config: timeout_sec=900.0.