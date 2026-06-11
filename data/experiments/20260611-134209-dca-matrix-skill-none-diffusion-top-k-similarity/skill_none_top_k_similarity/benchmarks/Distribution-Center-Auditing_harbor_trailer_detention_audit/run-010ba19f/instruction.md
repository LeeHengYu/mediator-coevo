# Task Instruction

Execute the following Python script to produce both deliverables. Before running, inspect the source workbook to understand its structure.

```bash
cd /root
python3 -c "
import subprocess, sys
for pkg in ['openpyxl', 'python-docx']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
"
```

Then run:

```python
import openpyxl
from openpyxl import Workbook, load_workbook
from docx import Document

# ── 1. Read source ──────────────────────────────────────────────
src = load_workbook('/root/Trailer_Detention_Log.xlsx')
src_ws = src.active
rows = list(src_ws.iter_rows(values_only=True))
headers = list(rows[0])
data = rows[1:]

print('Source headers:', headers)
print('Row count:', len(data))
print('Sample rows:', data[:3])

# Map column indices by name (flexible to minor naming differences)
def col_idx(name):
    for i, h in enumerate(headers):
        if h and name.lower() in str(h).lower():
            return i
    raise ValueError(f'Column {name!r} not found in {headers}')

i_load      = col_idx('Load ID')
i_carrier   = col_idx('Carrier')
i_allowed   = col_idx('Allowed Hold')
i_actual    = col_idx('Actual Hold')
i_seal_req  = col_idx('Seal Required')
i_seal_stat = col_idx('Seal Status')
i_yard      = col_idx('Yard')
i_dispatch  = col_idx('Dispatcher')

# ── 2. Build output workbook ────────────────────────────────────
wb = Workbook()

# --- RawData sheet ---
ws_raw = wb.active
ws_raw.title = 'RawData'
for r in rows:
    ws_raw.append(list(r))

# --- Formatted Data sheet ---
ws_fmt = wb.create_sheet('Formatted Data')
fmt_headers = [
    'Load ID', 'Carrier', 'Allowed Hold Hours', 'Actual Hold Hours',
    'Seal Required', 'Seal Status', 'Yard', 'Dispatcher',
    'Detention Overrun', 'Seal Error', 'Total Errors', 'Error Summary'
]
ws_fmt.append(fmt_headers)

formatted_rows = []
for row in data:
    load_id   = row[i_load]
    carrier   = row[i_carrier]
    allowed   = row[i_allowed]
    actual    = row[i_actual]
    seal_req  = row[i_seal_req]
    seal_stat = row[i_seal_stat]
    yard      = row[i_yard]
    dispatch  = row[i_dispatch]

    det_overrun = 1 if (actual is not None and allowed is not None and float(actual) > float(allowed)) else 0
    seal_err = 1 if (seal_req is not None and str(seal_req).strip().upper() == 'YES'
                     and (seal_stat is None or str(seal_stat).strip().upper() != 'VERIFIED')) else 0
    total_err = det_overrun + seal_err

    parts = []
    if det_overrun: parts.append('Detention Overrun')
    if seal_err:    parts.append('Seal Error')
    err_summary = ', '.join(parts) if parts else 'None'

    out_row = [load_id, carrier, allowed, actual, seal_req, seal_stat, yard, dispatch,
               det_overrun, seal_err, total_err, err_summary]
    ws_fmt.append(out_row)
    formatted_rows.append(out_row)

# --- Summary sheet ---
from collections import defaultdict
agg = defaultdict(lambda: [0, 0, 0])
for r in formatted_rows:
    key = (r[1], r[6])  # Carrier, Yard
    agg[key][0] += r[8]   # Detention Overrun
    agg[key][1] += r[9]   # Seal Error
    agg[key][2] += r[10]  # Total Errors

ws_sum = wb.create_sheet('Summary')
sum_headers = ['Carrier', 'Yard', 'Detention Overrun Errors', 'Seal Errors', 'Total Errors']
ws_sum.append(sum_headers)

sorted_keys = sorted([k for k in agg if agg[k][2] > 0], key=lambda k: (str(k[0]).lower(), str(k[1]).lower()))
tot_det = tot_seal = tot_all = 0
for k in sorted_keys:
    ws_sum.append([k[0], k[1], agg[k][0], agg[k][1], agg[k][2]])
    tot_det += agg[k][0]
    tot_seal += agg[k][1]
    tot_all += agg[k][2]

ws_sum.append(['Grand Total', '-', tot_det, tot_seal, tot_all])

wb.save('/root/Trailer_Detention_Audit.xlsx')
print('Audit workbook saved.')
print(f'Totals: Detention={tot_det}, Seal={tot_seal}, Total={tot_all}')

# ── 3. Find top carriers ────────────────────────────────────────
carrier_errs = defaultdict(int)
for r in formatted_rows:
    carrier_errs[r[1]] += r[10]
top_carriers = sorted(carrier_errs.items(), key=lambda x: -x[1])[:2]
print('Top carriers:', top_carriers)

# ── 4. Build Word brief ─────────────────────────────────────────
doc = Document()
doc.add_heading('Trailer Detention Audit – Executive Summary', level=1)

c1 = top_carriers[0][0] if len(top_carriers) > 0 else 'N/A'
c2 = top_carriers[1][0] if len(top_carriers) > 1 else 'N/A'

para = (
    f'This audit reviewed trailer detention compliance across all yard locations. '
    f'A Detention Overrun error is flagged when a trailer\u2019s Actual Hold Hours exceed its Allowed Hold Hours, '
    f'indicating the carrier held the trailer beyond the contractually permitted window. '
    f'A Seal Error is flagged when a seal is required (Seal Required = YES) but the Seal Status is not VERIFIED, '
    f'suggesting a potential security or chain-of-custody gap. '
    f'Across the dataset, {tot_det} Detention Overrun errors, {tot_seal} Seal Errors, '
    f'and {tot_all} Total Errors were identified. '
    f'{c1} and {c2} are high-priority carriers with the most frequent exceptions and should be targeted '
    f'for corrective action plans, including stricter scheduling windows and mandatory seal-verification checkpoints '
    f'before trailer release.'
)
doc.add_paragraph(para)
doc.save('/root/Trailer_Detention_Brief.docx')
print('Brief saved.')
```

After running, verify:
1. `ls -la /root/Trailer_Detention_Audit.xlsx /root/Trailer_Detention_Brief.docx` — both files exist.
2. Open the audit workbook and confirm sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
3. Confirm `Formatted Data` has 12 columns with the exact headers specified.
4. Confirm `Summary` ends with a `Grand Total` row and only includes groups with Total Errors > 0.
5. Confirm the Word doc contains the totals and at least two carrier names.

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