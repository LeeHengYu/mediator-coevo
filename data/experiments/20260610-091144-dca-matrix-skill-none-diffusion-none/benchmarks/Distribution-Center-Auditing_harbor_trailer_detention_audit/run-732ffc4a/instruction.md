# Task Instruction

Execute the following Python script in a single step to produce both deliverables.

```python
import openpyxl
from openpyxl import Workbook
from docx import Document
import os

# ── 1. Read source data ──────────────────────────────────────────────
src = openpyxl.load_workbook('/root/Trailer_Detention_Log.xlsx', data_only=True)
src_ws = src.active

# Read all rows (first row = header)
rows = []
for row in src_ws.iter_rows(min_row=1, values_only=True):
    rows.append(list(row))

header = rows[0]
data = rows[1:]

print(f'Source header: {header}')
print(f'Source row count (excl header): {len(data)}')
if data:
    print(f'Sample row: {data[0]}')

# ── 2. Identify column indices ───────────────────────────────────────
# Expected first 8 columns:
#   Load ID, Carrier, Allowed Hold Hours, Actual Hold Hours,
#   Seal Required, Seal Status, Yard, Dispatcher
# We locate them by name (case-insensitive, stripped) to be robust.

def find_col(header_list, name):
    name_lower = name.strip().lower()
    for i, h in enumerate(header_list):
        if h is not None and str(h).strip().lower() == name_lower:
            return i
    raise ValueError(f'Column "{name}" not found in {header_list}')

col_load_id       = find_col(header, 'Load ID')
col_carrier        = find_col(header, 'Carrier')
col_allowed        = find_col(header, 'Allowed Hold Hours')
col_actual         = find_col(header, 'Actual Hold Hours')
col_seal_req       = find_col(header, 'Seal Required')
col_seal_status    = find_col(header, 'Seal Status')
col_yard           = find_col(header, 'Yard')
col_dispatcher     = find_col(header, 'Dispatcher')

first8_indices = [col_load_id, col_carrier, col_allowed, col_actual,
                  col_seal_req, col_seal_status, col_yard, col_dispatcher]

print(f'Column indices: {first8_indices}')

# ── 3. Build Formatted Data rows ────────────────────────────────────
formatted_header = [
    'Load ID', 'Carrier', 'Allowed Hold Hours', 'Actual Hold Hours',
    'Seal Required', 'Seal Status', 'Yard', 'Dispatcher',
    'Detention Overrun', 'Seal Error', 'Total Errors', 'Error Summary'
]

formatted_rows = []
for r in data:
    base = [r[i] for i in first8_indices]

    # Detention Overrun
    try:
        actual = float(r[col_actual])
        allowed = float(r[col_allowed])
    except (TypeError, ValueError):
        actual = 0.0
        allowed = 0.0
    detention_overrun = 1 if actual > allowed else 0

    # Seal Error
    seal_req_val = str(r[col_seal_req]).strip().upper() if r[col_seal_req] is not None else ''
    seal_status_val = str(r[col_seal_status]).strip().upper() if r[col_seal_status] is not None else ''
    seal_error = 1 if seal_req_val == 'YES' and seal_status_val != 'VERIFIED' else 0

    total_errors = detention_overrun + seal_error

    # Error Summary
    parts = []
    if detention_overrun == 1:
        parts.append('Detention Overrun')
    if seal_error == 1:
        parts.append('Seal Error')
    error_summary = ', '.join(parts) if parts else 'None'

    formatted_rows.append(base + [detention_overrun, seal_error, total_errors, error_summary])

print(f'Formatted rows built: {len(formatted_rows)}')

# ── 4. Build Summary ────────────────────────────────────────────────
from collections import defaultdict

agg = defaultdict(lambda: [0, 0, 0])  # (carrier, yard) -> [det, seal, total]
for fr in formatted_rows:
    carrier = fr[1]
    yard = fr[6]
    det = fr[8]
    seal = fr[9]
    tot = fr[10]
    key = (str(carrier), str(yard))
    agg[key][0] += det
    agg[key][1] += seal
    agg[key][2] += tot

# Filter only groups with total > 0, sort by carrier then yard
summary_rows = []
for (carrier, yard), vals in agg.items():
    if vals[2] > 0:
        summary_rows.append([carrier, yard, vals[0], vals[1], vals[2]])

summary_rows.sort(key=lambda x: (x[0].lower(), x[1].lower()))

# Grand Total
grand_det = sum(v[0] for v in agg.values())
grand_seal = sum(v[1] for v in agg.values())
grand_total = sum(v[2] for v in agg.values())
summary_rows.append(['Grand Total', '-', grand_det, grand_seal, grand_total])

summary_header = ['Carrier', 'Yard', 'Detention Overrun Errors', 'Seal Errors', 'Total Errors']

print(f'Summary rows (incl Grand Total): {len(summary_rows)}')
print(f'Grand totals: det={grand_det}, seal={grand_seal}, total={grand_total}')

# ── 5. Write Excel workbook ─────────────────────────────────────────
wb = Workbook()

# RawData sheet
ws_raw = wb.active
ws_raw.title = 'RawData'
for r in rows:  # includes header
    ws_raw.append(r)

# Formatted Data sheet
ws_fmt = wb.create_sheet('Formatted Data')
ws_fmt.append(formatted_header)
for fr in formatted_rows:
    ws_fmt.append(fr)

# Summary sheet
ws_sum = wb.create_sheet('Summary')
ws_sum.append(summary_header)
for sr in summary_rows:
    ws_sum.append(sr)

wb.save('/root/Trailer_Detention_Audit.xlsx')
print('Excel saved.')

# ── 6. Identify top carriers for Word brief ─────────────────────────
# Exclude Grand Total row for carrier ranking
carrier_totals = defaultdict(int)
for sr in summary_rows[:-1]:  # skip Grand Total
    carrier_totals[sr[0]] += sr[4]

top_carriers = sorted(carrier_totals.items(), key=lambda x: -x[1])
top2 = [c[0] for c in top_carriers[:2]]
print(f'Top 2 carriers: {top2}')

# ── 7. Write Word brief ─────────────────────────────────────────────
doc = Document()
doc.add_heading('Trailer Detention Audit – Executive Summary', level=1)

p1 = (f'This audit evaluated trailer detention compliance across two checks. '
      f'A Detention Overrun is flagged when a trailer\'s Actual Hold Hours exceed '
      f'its Allowed Hold Hours, indicating the trailer was held beyond the contractual '
      f'free-time window. A Seal Error is flagged when a trailer that requires a seal '
      f'(Seal Required = YES) does not have a Seal Status of VERIFIED, suggesting '
      f'potential security or documentation lapses.')

p2 = (f'Across all records, the audit identified {grand_det} Detention Overrun '
      f'error(s), {grand_seal} Seal Error(s), and {grand_total} Total Error(s). '
      f'The carriers with the most frequent exceptions are {top2[0]} and {top2[1]}, '
      f'which should be prioritized for corrective action and root-cause analysis.')

p3 = (f'We recommend implementing automated gate-time alerts to notify dispatchers '
      f'before trailers exceed their allowed hold window, and conducting a quarterly '
      f'seal-verification audit for all carriers flagged with recurring Seal Errors.')

doc.add_paragraph(p1)
doc.add_paragraph(p2)
doc.add_paragraph(p3)

doc.save('/root/Trailer_Detention_Brief.docx')
print('Word doc saved.')
print('Done.')
```

After running the script, verify both output files exist:
```bash
ls -la /root/Trailer_Detention_Audit.xlsx /root/Trailer_Detention_Brief.docx
```

Then do a quick sanity check on the Excel output:
```python
import openpyxl
wb = openpyxl.load_workbook('/root/Trailer_Detention_Audit.xlsx')
print('Sheet names:', wb.sheetnames)
for name in wb.sheetnames:
    ws = wb[name]
    print(f'\n--- {name} ---')
    print(f'Rows: {ws.max_row}, Cols: {ws.max_column}')
    for row in ws.iter_rows(min_row=1, max_row=min(3, ws.max_row), values_only=True):
        print(row)
    if ws.max_row > 3:
        print('...')
        for row in ws.iter_rows(min_row=ws.max_row, max_row=ws.max_row, values_only=True):
            print(row)
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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=medium, tags=[excel, openpyxl, docx, audit, logistics].
Verifier config: timeout_sec=900.0.