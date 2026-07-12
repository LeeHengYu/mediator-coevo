# Task Instruction

Execute the following steps in order:

## 1. Inspect the source workbook
```python
import openpyxl
wb = openpyxl.load_workbook('/root/Timesheet_Submissions.xlsx')
print('Sheet names:', wb.sheetnames)

# Inspect Entries sheet
ws_entries = wb['Entries']
print('Entries headers:', [cell.value for cell in ws_entries[1]])
print('Entries row count (excluding header):', ws_entries.max_row - 1)
for row in ws_entries.iter_rows(min_row=2, max_row=min(6, ws_entries.max_row), values_only=True):
    print(row)

# Inspect BreakRules sheet
ws_rules = wb['BreakRules']
print('BreakRules headers:', [cell.value for cell in ws_rules[1]])
for row in ws_rules.iter_rows(min_row=2, max_row=ws_rules.max_row, values_only=True):
    print(row)
```

## 2. Build the output Excel and Word files

After inspecting the data and understanding the column positions, run a single Python script that:

### A. Read source data
- Read the `Entries` sheet into a list of dicts (preserving original column names and order).
- Read the `BreakRules` sheet into a dict keyed by Role, mapping to `{'min_break': ..., 'overtime_threshold': ...}`. Use the actual column names from the sheet (inspect first).

### B. Create `/root/Timesheet_Compliance_Audit.xlsx` with openpyxl

#### Sheet `RawData`
- Copy the `Entries` table exactly (headers + all data rows, same order).

#### Sheet `Formatted Data`
- Same row order as RawData.
- First 8 columns exactly: `Week Ending`, `Employee ID`, `Role`, `Hours Worked`, `Break Minutes`, `Approval Code`, `Project Code`, `Manager`.
  - If the source column names differ slightly, map them to these exact header strings.
- Add columns 9-12 with exact headers: `Break Deficit`, `Approval Missing`, `Total Errors`, `Error Summary`.
- Compute values as concrete numbers/strings (not formulas):
  - `Break Deficit` = 1 if `Break Minutes` < `Min Break Minutes` for that Role from BreakRules, else 0.
  - `Approval Missing` = 1 if `Hours Worked` > `Overtime Threshold` for that Role from BreakRules AND `Approval Code` is blank/None/empty, else 0.
  - `Total Errors` = Break Deficit + Approval Missing.
  - `Error Summary`: exactly one of `None`, `Break Deficit`, `Approval Missing`, `Break Deficit, Approval Missing` based on which flags are 1.
- Write all values as concrete Python int/str, not Excel formulas.

#### Sheet `Summary`
- Headers exactly: `Employee ID`, `Week Ending`, `Break Deficits`, `Approval Gaps`, `Total Errors`.
- Group `Formatted Data` rows by (Employee ID, Week Ending).
- Sum Break Deficit → `Break Deficits`, Approval Missing → `Approval Gaps`, Total Errors → `Total Errors` per group.
- Include only groups where Total Errors > 0.
- Sort by Employee ID ascending, then Week Ending ascending.
- Append a final row: `Grand Total`, `-`, sum of all Break Deficits, sum of all Approval Gaps, sum of all Total Errors.

Save to `/root/Timesheet_Compliance_Audit.xlsx`.

### C. Create `/root/Timesheet_Compliance_Brief.docx` with python-docx

- Install python-docx if needed: `pip install python-docx`
- Write an executive summary paragraph (3-6 sentences) that includes:
  1. Plain-language definition of Break Deficit check: "A Break Deficit is flagged when an employee's recorded break minutes fall below the minimum required for their role."
  2. Plain-language definition of Approval Missing check: "An Approval Missing flag is raised when an employee works beyond the overtime threshold for their role without a corresponding approval code."
  3. The computed grand totals: "Across all submissions, the audit identified X Break Deficits, Y Approval Gaps, and Z Total Errors."
  4. At least one actionable recommendation, e.g., "We recommend implementing automated pre-submission validation to catch break and approval deficiencies before timesheet finalization."
  5. Mention at least two specific Employee IDs with the highest error counts (pick the top 2 by total errors from the Summary data). Use the exact Employee ID strings, e.g., "Employees EMP-1234 and EMP-5678 had the most frequent exceptions and should be prioritized for corrective coaching."
- Save to `/root/Timesheet_Compliance_Brief.docx`.

## 3. Validate outputs

After creating both files, run validation:
```python
import openpyxl
from docx import Document

# Validate Excel
wb = openpyxl.load_workbook('/root/Timesheet_Compliance_Audit.xlsx')
print('Output sheets:', wb.sheetnames)
assert wb.sheetnames == ['RawData', 'Formatted Data', 'Summary'], f'Sheet names mismatch: {wb.sheetnames}'

ws_raw = wb['RawData']
print('RawData headers:', [c.value for c in ws_raw[1]])
print('RawData rows:', ws_raw.max_row - 1)

ws_fmt = wb['Formatted Data']
fmt_headers = [c.value for c in ws_fmt[1]]
print('Formatted Data headers:', fmt_headers)
assert fmt_headers[:8] == ['Week Ending', 'Employee ID', 'Role', 'Hours Worked', 'Break Minutes', 'Approval Code', 'Project Code', 'Manager']
assert fmt_headers[8:12] == ['Break Deficit', 'Approval Missing', 'Total Errors', 'Error Summary']
# Check a few data rows
for row in ws_fmt.iter_rows(min_row=2, max_row=min(5, ws_fmt.max_row), values_only=True):
    print(row)

ws_sum = wb['Summary']
sum_headers = [c.value for c in ws_sum[1]]
print('Summary headers:', sum_headers)
assert sum_headers == ['Employee ID', 'Week Ending', 'Break Deficits', 'Approval Gaps', 'Total Errors']
# Check last row is Grand Total
last_row = [c.value for c in ws_sum[ws_sum.max_row]]
print('Last Summary row:', last_row)
assert last_row[0] == 'Grand Total'
assert last_row[1] == '-'
print('Summary row count (excl header):', ws_sum.max_row - 1)

# Validate Word doc
doc = Document('/root/Timesheet_Compliance_Brief.docx')
full_text = ' '.join([p.text for p in doc.paragraphs])
print('Word doc text length:', len(full_text))
assert 'Break Deficit' in full_text
assert 'Approval Missing' in full_text
# Check that at least 2 employee IDs are mentioned
print('Word doc preview:', full_text[:500])
print('VALIDATION PASSED')
```

## 4. If test_output.py exists, run it
```bash
if [ -f /root/test_output.py ]; then cd /root && python -m pytest test_output.py -v; fi
```

## Key Warnings
- When checking if Approval Code is blank, treat None, empty string '', and whitespace-only as blank.
- Use the BreakRules thresholds dynamically (lookup by role), do not hardcode numeric values.
- The `Error Summary` string must use exactly `None` (not empty string), `Break Deficit`, `Approval Missing`, or `Break Deficit, Approval Missing` — match these strings exactly including comma and space.
- For the Word document, make sure to include the actual Employee ID strings (e.g., `EMP-1234`) so the verifier can find them. Pick the top 2 employees by total errors from the Summary sheet. This is critical — a prior similar task failed because specific identifiers were not mentioned in the Word doc.
- The Grand Total row values must be integers (sums), not strings.

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