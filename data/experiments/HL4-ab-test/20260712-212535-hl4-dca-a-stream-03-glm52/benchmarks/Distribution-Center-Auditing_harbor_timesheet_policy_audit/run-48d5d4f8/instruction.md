# Task Instruction

Execute the following steps in a single Python script to produce the two deliverables.

## Step 0 – Inspect the source workbook
```python
import openpyxl
wb = openpyxl.load_workbook('/root/Timesheet_Submissions.xlsx')
print('Sheet names:', wb.sheetnames)
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'\n--- {sn} (rows={ws.max_row}, cols={ws.max_column}) ---')
    for r in ws.iter_rows(min_row=1, max_row=min(5, ws.max_row), values_only=True):
        print(r)
wb.close()
```
Run this first and read the output carefully. Note:
- The exact column names in `Entries` (they must map to: Week Ending, Employee ID, Role, Hours Worked, Break Minutes, Approval Code, Project Code, Manager).
- The exact column names and values in `BreakRules` (especially the columns for Role, Min Break Minutes, and Overtime Threshold).

## Step 1 – Build the audit Excel file

After inspecting, write and run a script (adjust column names to match what you actually see):

```python
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from copy import copy

# ---- Load source data ----
entries = pd.read_excel('/root/Timesheet_Submissions.xlsx', sheet_name='Entries')
break_rules = pd.read_excel('/root/Timesheet_Submissions.xlsx', sheet_name='BreakRules')

print('Entries columns:', list(entries.columns))
print('BreakRules columns:', list(break_rules.columns))
print('BreakRules data:')
print(break_rules)
print('Entries head:')
print(entries.head())

# ---- Rename source columns to canonical names if needed ----
# Map source column names → canonical names used in the task.
# IMPORTANT: adjust the left-hand side keys to match the ACTUAL column names you saw in Step 0.
# Example (adjust as needed):
# entries.rename(columns={'SourceName': 'CanonicalName'}, inplace=True)

# The first 8 columns in output must be exactly:
required_cols = ['Week Ending','Employee ID','Role','Hours Worked','Break Minutes',
                 'Approval Code','Project Code','Manager']

# ---- Build BreakRules lookup (Role → min_break, ot_threshold) ----
# Adjust key names to match actual BreakRules columns.
rule_map = {}
for _, row in break_rules.iterrows():
    role = row.iloc[0]  # first col should be Role
    min_break = row.iloc[1]  # Min Break Minutes
    ot_thresh = row.iloc[2]  # Overtime Threshold
    # Also try by column name; print to verify
    rule_map[str(role).strip()] = {'min_break': float(min_break), 'ot_thresh': float(ot_thresh)}
print('rule_map:', rule_map)

# ---- Compute flags ----
def compute_flags(row):
    role = str(row['Role']).strip()
    rules = rule_map.get(role, {})
    min_break = rules.get('min_break', 0)
    ot_thresh = rules.get('ot_thresh', float('inf'))

    brk = row['Break Minutes']
    hrs = row['Hours Worked']
    appr = row['Approval Code']

    break_deficit = 1 if (pd.notna(brk) and brk < min_break) else 0
    # Approval Missing: hours > threshold AND approval code is blank/NaN
    approval_blank = (pd.isna(appr) or str(appr).strip() == '')
    approval_missing = 1 if (pd.notna(hrs) and hrs > ot_thresh and approval_blank) else 0

    total_errors = break_deficit + approval_missing

    parts = []
    if break_deficit:
        parts.append('Break Deficit')
    if approval_missing:
        parts.append('Approval Missing')
    error_summary = ', '.join(parts) if parts else 'None'

    return pd.Series([break_deficit, approval_missing, total_errors, error_summary],
                     index=['Break Deficit','Approval Missing','Total Errors','Error Summary'])

formatted = entries[required_cols].copy()
flags = formatted.apply(compute_flags, axis=1)
formatted = pd.concat([formatted, flags], axis=1)

print('Formatted head:')
print(formatted.head(10))
print('Total Break Deficits:', formatted['Break Deficit'].sum())
print('Total Approval Missing:', formatted['Approval Missing'].sum())
print('Total Errors:', formatted['Total Errors'].sum())

# ---- Summary sheet ----
grouped = formatted.groupby(['Employee ID','Week Ending']).agg(
    **{'Break Deficits': ('Break Deficit','sum'),
       'Approval Gaps': ('Approval Missing','sum'),
       'Total Errors': ('Total Errors','sum')}
).reset_index()

# Keep only groups with Total Errors > 0
summary = grouped[grouped['Total Errors'] > 0].copy()
summary.sort_values(['Employee ID','Week Ending'], inplace=True)
summary.reset_index(drop=True, inplace=True)

# Grand Total row
grand = pd.DataFrame([{
    'Employee ID': 'Grand Total',
    'Week Ending': '-',
    'Break Deficits': summary['Break Deficits'].sum(),
    'Approval Gaps': summary['Approval Gaps'].sum(),
    'Total Errors': summary['Total Errors'].sum()
}])
summary = pd.concat([summary, grand], ignore_index=True)

print('Summary:')
print(summary)

# ---- Write Excel workbook ----
out_path = '/root/Timesheet_Compliance_Audit.xlsx'
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    entries[required_cols[:len(entries.columns)]].to_excel(writer, sheet_name='RawData', index=False)
    # Actually write the raw entries exactly as-is (all original columns)
    # Re-read: task says "Copy the Entries table exactly" so use original entries df
    # Overwrite:

# Redo write properly:
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    entries.to_excel(writer, sheet_name='RawData', index=False)
    formatted.to_excel(writer, sheet_name='Formatted Data', index=False)
    summary.to_excel(writer, sheet_name='Summary', index=False)

print('Excel written to', out_path)
```

After writing, verify by re-reading:
```python
import openpyxl
wb = openpyxl.load_workbook(out_path)
print('Output sheets:', wb.sheetnames)
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'\n--- {sn} ---')
    for i, r in enumerate(ws.iter_rows(min_row=1, max_row=min(5, ws.max_row), values_only=True)):
        print(r)
    print(f'  total rows (incl header): {ws.max_row}')
wb.close()
```

Confirm:
- Sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
- `RawData` has the same data as `Entries`.
- `Formatted Data` columns 1-8 match required names, columns 9-12 are `Break Deficit`, `Approval Missing`, `Total Errors`, `Error Summary`.
- `Summary` columns are `Employee ID`, `Week Ending`, `Break Deficits`, `Approval Gaps`, `Total Errors`.
- Last row of Summary has `Grand Total`.
- Values are concrete numbers/strings, not formulas.

## Step 2 – Build the Word document

From the computed data, identify the top employee IDs with the most total errors. Then write the brief:

```python
from docx import Document

# Identify top 2 employees by total errors
emp_errors = formatted.groupby('Employee ID')['Total Errors'].sum().reset_index()
emp_errors.sort_values('Total Errors', ascending=False, inplace=True)
top_emps = emp_errors[emp_errors['Total Errors'] > 0].head(2)
print('Top employees:', top_emps)

total_bd = int(formatted['Break Deficit'].sum())
total_am = int(formatted['Approval Missing'].sum())
total_err = int(formatted['Total Errors'].sum())

emp1 = str(top_emps.iloc[0]['Employee ID'])
emp2 = str(top_emps.iloc[1]['Employee ID'])

doc = Document()
doc.add_heading('Timesheet Compliance Brief', level=1)

para_text = (
    f'This report summarizes the weekly timesheet compliance audit. '
    f'Two compliance checks were applied to every submission: '
    f'"Break Deficit" flags entries where the recorded break minutes fall below the minimum '
    f'required for the employee\'s role, and "Approval Missing" flags entries where hours worked '
    f'exceed the overtime threshold for the role but no approval code was provided. '
    f'Across all submissions, the audit identified {total_bd} Break Deficits, '
    f'{total_am} Approval Gaps, and {total_err} Total Errors. '
    f'Employees {emp1} and {emp2} were identified as high-priority cases due to their '
    f'frequent exceptions. '
    f'It is recommended that managers conduct targeted reviews of break compliance '
    f'and ensure overtime approval codes are submitted before the weekly deadline.'
)

doc.add_paragraph(para_text)

word_path = '/root/Timesheet_Compliance_Brief.docx'
doc.save(word_path)
print('Word document saved to', word_path)
```

After saving, verify:
```python
from docx import Document
doc = Document(word_path)
for p in doc.paragraphs:
    print(p.text)
```

Confirm the text includes:
- Definitions of both checks.
- The numeric totals for Break Deficits, Approval Gaps, Total Errors.
- At least one recommendation.
- At least two employee IDs mentioned explicitly.

## Step 3 – Final checks

1. Confirm both files exist: `ls -la /root/Timesheet_Compliance_Audit.xlsx /root/Timesheet_Compliance_Brief.docx`
2. Re-open the Excel file and print the Summary sheet completely to verify the Grand Total row and sorting.
3. Re-open the Word file and print all paragraph text to verify content.

## Critical Notes
- In Step 0, read the ACTUAL column names from the source workbook. Do NOT assume they match the canonical names. Map them accordingly.
- For `Approval Code` blank check: treat both NaN and empty string as blank.
- The `RawData` sheet must be an exact copy of `Entries` — all original columns, same order.
- The `Formatted Data` sheet must have exactly 12 columns with the exact header names specified.
- The `Summary` sheet must have exactly 5 columns with the exact header names specified.
- The Word document must mention employee IDs in a format that a simple substring search can find (e.g., include the raw Employee ID value like `EMP001`).
- Write concrete values (int 0 or 1, strings), not Excel formulas.
- Use thresholds from `BreakRules` dynamically, not hardcoded by role name.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=hard, tags=[excel, openpyxl, docx, audit, timesheet].
Verifier config: timeout_sec=900.0.