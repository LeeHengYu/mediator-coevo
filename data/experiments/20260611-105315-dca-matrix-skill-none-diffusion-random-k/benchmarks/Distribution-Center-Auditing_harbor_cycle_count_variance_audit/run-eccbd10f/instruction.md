# Task Instruction

Execute the following Python script in a single step to produce both deliverables. Read the script carefully before running it.

```python
import shutil
import pandas as pd
import openpyxl
from docx import Document

# ── 1. Read source files ──────────────────────────────────────────────
plan_df = pd.read_excel('/root/Cycle_Plan.xlsx')
event_df = pd.read_excel('/root/Count_Event_Log.xlsx')

# ── 2. Copy template to preserve Overview sheet ──────────────────────
shutil.copy('/root/Cycle_Template.xlsx', '/root/Cycle_Count_Variance_Audit.xlsx')

# ── 3. Process Count_Event_Log to get latest FINAL counts ────────────
# Filter to FINAL events with non-blank keys and non-blank Count Qty
final_events = event_df.copy()
# Standardise Event Type column name
et_col = [c for c in final_events.columns if 'event' in c.lower() and 'type' in c.lower()]
if et_col:
    et_col = et_col[0]
else:
    et_col = 'Event Type'

# Identify key columns
fac_col_ev = [c for c in final_events.columns if 'facility' in c.lower()][0]
sid_col_ev = [c for c in final_events.columns if 'session' in c.lower()][0]
bid_col_ev = [c for c in final_events.columns if 'bin' in c.lower()][0]
cqty_col = [c for c in final_events.columns if 'count' in c.lower() and 'qty' in c.lower()][0]

final_events = final_events[final_events[et_col].astype(str).str.strip().str.upper() == 'FINAL'].copy()
# Drop rows with blank keys or blank Count Qty
for col in [fac_col_ev, sid_col_ev, bid_col_ev, cqty_col]:
    final_events = final_events[final_events[col].notna()]
    final_events = final_events[final_events[col].astype(str).str.strip() != '']

# Keep only the latest row per (Facility, Session ID, Bin ID)
final_events = final_events.reset_index(drop=True)
final_events['_order'] = final_events.index
final_events = final_events.sort_values('_order').groupby(
    [fac_col_ev, sid_col_ev, bid_col_ev], as_index=False
).last()

# Build lookup dict: (facility, session, bin) -> count_qty
lookup = {}
for _, row in final_events.iterrows():
    key = (str(row[fac_col_ev]).strip(), str(row[sid_col_ev]).strip(), str(row[bid_col_ev]).strip())
    lookup[key] = float(row[cqty_col])

# ── 4. Build Formatted Data from plan_df ─────────────────────────────
# Identify plan columns
plan_cols = list(plan_df.columns)
fac_col = [c for c in plan_cols if 'facility' in c.lower()][0]
sid_col = [c for c in plan_cols if 'session' in c.lower()][0]
bid_col = [c for c in plan_cols if 'bin' in c.lower()][0]
pid_col = [c for c in plan_cols if 'product' in c.lower()][0]
eqty_col = [c for c in plan_cols if 'expected' in c.lower()][0]
avar_col = [c for c in plan_cols if 'variance' in c.lower()][0]
appr_col = [c for c in plan_cols if 'approval' in c.lower()][0]

formatted = plan_df[[fac_col, sid_col, bid_col, pid_col, eqty_col, avar_col, appr_col]].copy()
formatted.columns = ['Facility', 'Session ID', 'Bin ID', 'Product ID',
                     'Expected Qty', 'Allowed Variance', 'Approval Needed']

missing_list = []
approval_gap_list = []
total_errors_list = []
error_summary_list = []

for _, row in formatted.iterrows():
    key = (str(row['Facility']).strip(), str(row['Session ID']).strip(), str(row['Bin ID']).strip())
    has_final = key in lookup
    
    missing = 0 if has_final else 1
    
    ag = 0
    if has_final:
        appr = str(row['Approval Needed']).strip().upper()
        if appr == 'YES':
            count_qty = lookup[key]
            expected = float(row['Expected Qty'])
            allowed = float(row['Allowed Variance'])
            if abs(expected - count_qty) > allowed:
                ag = 1
    
    total = missing + ag
    
    parts = []
    if missing == 1:
        parts.append('Missing Final Count')
    if ag == 1:
        parts.append('Approval Gap')
    summary = ', '.join(parts) if parts else 'None'
    
    missing_list.append(missing)
    approval_gap_list.append(ag)
    total_errors_list.append(total)
    error_summary_list.append(summary)

formatted['Missing Final Count'] = missing_list
formatted['Approval Gap'] = approval_gap_list
formatted['Total Errors'] = total_errors_list
formatted['Error Summary'] = error_summary_list

# ── 5. Build Summary ─────────────────────────────────────────────────
summary_df = formatted.groupby(['Facility', 'Session ID'], as_index=False).agg(
    **{'Missing Final Counts': ('Missing Final Count', 'sum'),
       'Approval Gaps': ('Approval Gap', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
)
summary_df = summary_df[summary_df['Total Errors'] > 0].copy()
summary_df = summary_df.sort_values(['Facility', 'Session ID']).reset_index(drop=True)

# Grand Total row
grand = pd.DataFrame([{
    'Facility': 'Grand Total',
    'Session ID': '-',
    'Missing Final Counts': formatted['Missing Final Count'].sum(),
    'Approval Gaps': formatted['Approval Gap'].sum(),
    'Total Errors': formatted['Total Errors'].sum()
}])
summary_with_grand = pd.concat([summary_df, grand], ignore_index=True)

# ── 6. Write Excel using overlay mode to preserve Overview ───────────
with pd.ExcelWriter('/root/Cycle_Count_Variance_Audit.xlsx', engine='openpyxl',
                    mode='a', if_sheet_exists='replace') as writer:
    plan_df.to_excel(writer, sheet_name='RawData', index=False)
    formatted.to_excel(writer, sheet_name='Formatted Data', index=False)
    summary_with_grand.to_excel(writer, sheet_name='Summary', index=False)

# Verify sheet names
wb = openpyxl.load_workbook('/root/Cycle_Count_Variance_Audit.xlsx')
print('Sheet names:', wb.sheetnames)
wb.close()

# ── 7. Identify top facility-session pairs for Word doc ──────────────
# Use summary_df (without grand total) sorted by Total Errors descending
top_pairs = summary_df.sort_values('Total Errors', ascending=False).head(4)
top_mentions = []
for _, r in top_pairs.iterrows():
    fac = str(r['Facility']).strip()
    sid = str(r['Session ID']).strip()
    te = int(r['Total Errors'])
    top_mentions.append(f"{fac}-{sid} ({te} errors)")

total_missing = int(formatted['Missing Final Count'].sum())
total_ag = int(formatted['Approval Gap'].sum())
total_te = int(formatted['Total Errors'].sum())

print(f'Total Missing Final Counts: {total_missing}')
print(f'Total Approval Gaps: {total_ag}')
print(f'Total Errors: {total_te}')
print(f'Top pairs: {top_mentions}')

# ── 8. Create Word document ──────────────────────────────────────────
doc = Document()
doc.add_heading('Cycle Count Variance Audit – Executive Brief', level=1)

para1 = (f'This audit evaluated cycle-count accuracy across distribution-center '
         f'facilities by applying two key checks. '
         f'A "Missing Final Count" flags any planned bin that lacks a finalized '
         f'recount event, indicating the count was never completed. '
         f'An "Approval Gap" flags bins where a final count exists, approval is '
         f'required, and the absolute variance between the expected and counted '
         f'quantities exceeds the allowed threshold—meaning the discrepancy was '
         f'not properly reviewed or approved.')
doc.add_paragraph(para1)

para2 = (f'Across the dataset the audit identified {total_missing} Missing Final '
         f'Counts, {total_ag} Approval Gaps, and {total_te} Total Errors. '
         f'High-priority facility-session combinations with the most frequent '
         f'exceptions include {", ".join(top_mentions[:2])}'
         + (f', as well as {", ".join(top_mentions[2:])}' if len(top_mentions) > 2 else '')
         + '.')
doc.add_paragraph(para2)

para3 = ('We recommend that operations management prioritize recounts for bins '
         'flagged as Missing Final Count and institute a secondary review process '
         'for sessions with Approval Gaps. Facilities with the highest error '
         'concentrations should receive targeted training and tighter SLA '
         'enforcement for cycle-count completion.')
doc.add_paragraph(para3)

doc.save('/root/Cycle_Count_Variance_Brief.docx')
print('Word document saved.')
print('Done.')
```

After running the script, verify:
1. Open `/root/Cycle_Count_Variance_Audit.xlsx` and confirm it has exactly sheets: Overview, RawData, Formatted Data, Summary.
2. Check that the Summary sheet has a Grand Total row and only rows with Total Errors > 0.
3. Open `/root/Cycle_Count_Variance_Brief.docx` and confirm it mentions at least two facility-session pairs (in 'Facility-SessionID' format), the computed totals, definitions of both checks, and a recommendation.
4. Print the first few rows of Formatted Data and Summary for visual confirmation.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=expert, tags=[excel, openpyxl, docx, audit, inventory].
Verifier config: timeout_sec=900.0.