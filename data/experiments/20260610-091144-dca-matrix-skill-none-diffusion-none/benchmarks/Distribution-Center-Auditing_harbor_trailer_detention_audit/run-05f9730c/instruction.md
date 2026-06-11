# Task Instruction

Execute the following Python script to produce both deliverables. Before running, inspect the source file to understand its structure.

```bash
cd /root && python3 << 'PYEOF'
import openpyxl
from openpyxl import load_workbook, Workbook
from docx import Document
from collections import defaultdict

# ── 1. Read source data ──────────────────────────────────────────────
src = load_workbook('/root/Trailer_Detention_Log.xlsx', data_only=True)
src_ws = src.active
rows = list(src_ws.iter_rows(min_row=1, values_only=True))
header = list(rows[0])
data = [list(r) for r in rows[1:]]
print(f'Source columns: {header}')
print(f'Source row count (excl header): {len(data)}')

# Identify column indices by name (case-insensitive match)
def col_idx(name):
    for i, h in enumerate(header):
        if str(h).strip().lower() == name.strip().lower():
            return i
    raise ValueError(f'Column not found: {name}')

i_load      = col_idx('Load ID')
i_carrier   = col_idx('Carrier')
i_allowed   = col_idx('Allowed Hold Hours')
i_actual    = col_idx('Actual Hold Hours')
i_seal_req  = col_idx('Seal Required')
i_seal_stat = col_idx('Seal Status')
i_yard      = col_idx('Yard')
i_dispatch  = col_idx('Dispatcher')

# The 8 columns to keep, in required order
keep_indices = [i_load, i_carrier, i_allowed, i_actual, i_seal_req, i_seal_stat, i_yard, i_dispatch]

# ── 2. Build audit workbook ──────────────────────────────────────────
wb = Workbook()

# --- RawData sheet ---
ws_raw = wb.active
ws_raw.title = 'RawData'
ws_raw.append(header)
for row in data:
    ws_raw.append(row)

# --- Formatted Data sheet ---
ws_fmt = wb.create_sheet('Formatted Data')
fmt_header = ['Load ID', 'Carrier', 'Allowed Hold Hours', 'Actual Hold Hours',
              'Seal Required', 'Seal Status', 'Yard', 'Dispatcher',
              'Detention Overrun', 'Seal Error', 'Total Errors', 'Error Summary']
ws_fmt.append(fmt_header)

formatted_rows = []
for row in data:
    base = [row[i] for i in keep_indices]
    allowed = float(row[i_allowed]) if row[i_allowed] is not None else 0
    actual  = float(row[i_actual])  if row[i_actual]  is not None else 0
    seal_req  = str(row[i_seal_req]).strip().upper()  if row[i_seal_req]  is not None else ''
    seal_stat = str(row[i_seal_stat]).strip().upper() if row[i_seal_stat] is not None else ''

    det_overrun = 1 if actual > allowed else 0
    seal_error  = 1 if seal_req == 'YES' and seal_stat != 'VERIFIED' else 0
    total_err   = det_overrun + seal_error

    if total_err == 0:
        summary = 'None'
    elif det_overrun == 1 and seal_error == 1:
        summary = 'Detention Overrun, Seal Error'
    elif det_overrun == 1:
        summary = 'Detention Overrun'
    else:
        summary = 'Seal Error'

    full_row = base + [det_overrun, seal_error, total_err, summary]
    ws_fmt.append(full_row)
    formatted_rows.append(full_row)

# --- Summary sheet ---
ws_sum = wb.create_sheet('Summary')
sum_header = ['Carrier', 'Yard', 'Detention Overrun Errors', 'Seal Errors', 'Total Errors']
ws_sum.append(sum_header)

# Aggregate by (Carrier, Yard)
agg = defaultdict(lambda: [0, 0, 0])
for fr in formatted_rows:
    carrier = fr[1]
    yard    = fr[6]
    det_o   = fr[8]
    seal_e  = fr[9]
    tot_e   = fr[10]
    agg[(carrier, yard)][0] += det_o
    agg[(carrier, yard)][1] += seal_e
    agg[(carrier, yard)][2] += tot_e

# Filter to groups with Total Errors > 0, sort by Carrier asc then Yard asc
filtered = [(k, v) for k, v in agg.items() if v[2] > 0]
filtered.sort(key=lambda x: (str(x[0][0]).lower(), str(x[0][1]).lower()))

grand_det = 0
grand_seal = 0
grand_tot = 0
for (carrier, yard), vals in filtered:
    ws_sum.append([carrier, yard, vals[0], vals[1], vals[2]])
    grand_det  += vals[0]
    grand_seal += vals[1]
    grand_tot  += vals[2]

ws_sum.append(['Grand Total', '-', grand_det, grand_seal, grand_tot])

wb.save('/root/Trailer_Detention_Audit.xlsx')
print('Audit workbook saved.')
print(f'Grand totals — Detention Overrun: {grand_det}, Seal Errors: {grand_seal}, Total: {grand_tot}')

# ── 3. Identify top carriers for the brief ───────────────────────────
carrier_errors = defaultdict(int)
for fr in formatted_rows:
    carrier = fr[1]
    tot_e   = fr[10]
    carrier_errors[carrier] += tot_e

top_carriers = sorted(carrier_errors.items(), key=lambda x: -x[1])
print(f'Carrier error ranking: {top_carriers}')
top2 = [c for c, _ in top_carriers[:2] if _ > 0]

# ── 4. Word brief ────────────────────────────────────────────────────
doc = Document()
doc.add_heading('Trailer Detention Audit – Executive Summary', level=1)

lines = []
lines.append(
    'This audit examines trailer detention compliance across the yard network using two checks: '
    'a Detention Overrun flag, which is raised when a trailer\'s Actual Hold Hours exceed its '
    'Allowed Hold Hours, and a Seal Error flag, which is raised when a trailer that requires '
    'a seal (Seal Required = YES) does not have a Seal Status of VERIFIED.'
)
lines.append(
    f'Across the dataset, {grand_det} Detention Overrun error(s), {grand_seal} Seal Error(s), '
    f'and {grand_tot} Total Error(s) were identified.'
)
if len(top2) >= 2:
    lines.append(
        f'The carriers with the most frequent exceptions are {top2[0]} and {top2[1]}, '
        'which should be prioritized for corrective action and closer monitoring.'
    )
elif len(top2) == 1:
    lines.append(
        f'The carrier with the most frequent exceptions is {top2[0]}, '
        'which should be prioritized for corrective action.'
    )
lines.append(
    'We recommend implementing automated gate-time alerts to prevent detention overruns '
    'and adding a mandatory seal-verification checkpoint before trailers are released from the yard.'
)

doc.add_paragraph(' '.join(lines))
doc.save('/root/Trailer_Detention_Brief.docx')
print('Word brief saved.')
PYEOF
```

After the script completes, verify the outputs:

```bash
python3 << 'VEOF'
from openpyxl import load_workbook
from docx import Document

# Verify Excel
wb = load_workbook('/root/Trailer_Detention_Audit.xlsx')
print('Sheet names:', wb.sheetnames)
assert wb.sheetnames == ['RawData', 'Formatted Data', 'Summary'], f'Sheet names mismatch: {wb.sheetnames}'

ws_raw = wb['RawData']
ws_fmt = wb['Formatted Data']
ws_sum = wb['Summary']

print(f'RawData rows: {ws_raw.max_row}')
print(f'Formatted Data rows: {ws_fmt.max_row}')
print(f'Summary rows: {ws_sum.max_row}')

# Check Formatted Data header
fmt_h = [c.value for c in ws_fmt[1]]
expected_h = ['Load ID', 'Carrier', 'Allowed Hold Hours', 'Actual Hold Hours',
              'Seal Required', 'Seal Status', 'Yard', 'Dispatcher',
              'Detention Overrun', 'Seal Error', 'Total Errors', 'Error Summary']
assert fmt_h == expected_h, f'Formatted Data header mismatch: {fmt_h}'

# Check Summary header
sum_h = [c.value for c in ws_sum[1]]
expected_sh = ['Carrier', 'Yard', 'Detention Overrun Errors', 'Seal Errors', 'Total Errors']
assert sum_h == expected_sh, f'Summary header mismatch: {sum_h}'

# Check last row of Summary is Grand Total
last_row = [c.value for c in ws_sum[ws_sum.max_row]]
assert last_row[0] == 'Grand Total', f'Last summary row not Grand Total: {last_row}'
assert last_row[1] == '-', f'Grand Total yard not dash: {last_row}'
print(f'Grand Total row: {last_row}')

# Verify Word
doc = Document('/root/Trailer_Detention_Brief.docx')
text = ' '.join([p.text for p in doc.paragraphs])
assert 'Detention Overrun' in text
assert 'Seal Error' in text
assert str(last_row[4]) in text  # Total Errors number
print('Word brief verified.')
print('ALL CHECKS PASSED')
VEOF
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