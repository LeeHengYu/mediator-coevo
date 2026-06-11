# Task Instruction

Execute the following steps in order to produce the two deliverables.

## Step 1 – Inspect the source workbook

```python
import openpyxl
wb = openpyxl.load_workbook('/root/Timesheet_Submissions.xlsx')
print('Sheet names:', wb.sheetnames)

# Print Entries header + first 5 rows
ws_entries = wb['Entries']
for i, row in enumerate(ws_entries.iter_rows(values_only=True)):
    print(row)
    if i >= 5:
        break

# Print BreakRules header + all rows
ws_rules = wb['BreakRules']
for row in ws_rules.iter_rows(values_only=True):
    print(row)

wb.close()
```

Read the output carefully before proceeding. Note the exact column headers, data types, and the role-based thresholds.

## Step 2 – Build the output files with a single Python script

Write and run a Python script `/root/build_audit.py` that does everything below. Use `openpyxl` for Excel and `python-docx` for Word.

### 2a – Read source data
- Load `Entries` into a list of dicts (use the first row as headers).
- Load `BreakRules` into a dict keyed by Role → {"Min Break Minutes": …, "Overtime Threshold": …} (use the exact column names from the sheet; adapt if the header names differ from these guesses).

### 2b – Build `Timesheet_Compliance_Audit.xlsx`

#### Sheet `RawData`
- Copy the `Entries` data exactly (same headers, same row order, same values).

#### Sheet `Formatted Data`
- Same rows, same order.
- First 8 columns exactly: Week Ending, Employee ID, Role, Hours Worked, Break Minutes, Approval Code, Project Code, Manager.
- Column 9 `Break Deficit`: 1 if Break Minutes < Min Break Minutes for that Role, else 0.
- Column 10 `Approval Missing`: 1 if Hours Worked > Overtime Threshold for that Role AND Approval Code is blank/None/empty string, else 0.
- Column 11 `Total Errors`: Break Deficit + Approval Missing.
- Column 12 `Error Summary`: exactly one of `None`, `Break Deficit`, `Approval Missing`, `Break Deficit, Approval Missing` (use the string `"None"` when Total Errors == 0).
- **Write concrete values (int 0/1 and strings), not formulas.**

#### Sheet `Summary`
- Headers: Employee ID, Week Ending, Break Deficits, Approval Gaps, Total Errors.
- Group `Formatted Data` rows by (Employee ID, Week Ending). Sum Break Deficit → Break Deficits, Approval Missing → Approval Gaps, Total Errors → Total Errors.
- Include only groups where Total Errors > 0.
- Sort by Employee ID ascending then Week Ending ascending.
- Append a Grand Total row: Employee ID = "Grand Total", Week Ending = "-", and the remaining columns are the dataset-wide sums of Break Deficits, Approval Gaps, Total Errors.

### 2c – Build `Timesheet_Compliance_Brief.docx`

Create a Word document with a heading "Timesheet Compliance Brief" and one paragraph (3-6 sentences) that satisfies ALL of the following:

1. Uses the **exact phrases** "Break Deficit" and "Approval Missing" (both capitalised as shown) when defining the two checks. For example: "A Break Deficit flags any entry where the recorded break time falls below the minimum required for the employee's role. An Approval Missing flag is raised when hours exceed the overtime threshold for the role but no approval code is recorded."
2. Also include the phrases in **lowercase** form at least once each ("break deficit" and "approval missing") so that a case-insensitive search for 'approval missing' will match. You can do this naturally, e.g., "The most common issue was break deficit, followed by approval missing."
3. States the computed totals: "The audit identified X Break Deficits, Y Approval Gaps, and Z Total Errors across all submissions." (replace X, Y, Z with actual numbers from the Grand Total row).
4. Names at least two Employee IDs that have the highest Total Errors counts and labels them high-priority.
5. Includes at least one actionable recommendation.

### 2d – Save both files
- `/root/Timesheet_Compliance_Audit.xlsx`
- `/root/Timesheet_Compliance_Brief.docx`

## Step 3 – Validate

After the script finishes, run these checks:

```python
import openpyxl
from docx import Document

# Check Excel
wb = openpyxl.load_workbook('/root/Timesheet_Compliance_Audit.xlsx')
print('Sheets:', wb.sheetnames)
assert 'RawData' in wb.sheetnames
assert 'Formatted Data' in wb.sheetnames
assert 'Summary' in wb.sheetnames

ws = wb['Formatted Data']
headers = [c.value for c in ws[1]]
print('Formatted Data headers:', headers)
assert headers[8] == 'Break Deficit'
assert headers[9] == 'Approval Missing'
assert headers[10] == 'Total Errors'
assert headers[11] == 'Error Summary'

ws_sum = wb['Summary']
sum_headers = [c.value for c in ws_sum[1]]
print('Summary headers:', sum_headers)
# Print last row (Grand Total)
for row in ws_sum.iter_rows(values_only=True):
    last = row
print('Grand Total row:', last)

wb.close()

# Check Word
doc = Document('/root/Timesheet_Compliance_Brief.docx')
text = '\n'.join(p.text for p in doc.paragraphs).lower()
print('--- Word text (lower) ---')
print(text)
assert 'break deficit' in text, 'Missing phrase: break deficit'
assert 'approval missing' in text, 'Missing phrase: approval missing'
print('All quick checks passed.')
```

If any assertion fails, diagnose and fix before finishing.

## Key Reminders
- The verifier checks for the **exact lowercase string** `'approval missing'` in the Word document text. Make sure the phrase appears verbatim.
- Use thresholds from `BreakRules`, not hardcoded values.
- Approval Code is considered blank if it is None, empty string, or whitespace-only.
- Write concrete values in columns 9-12 of `Formatted Data` (no Excel formulas).
- Keep filenames and sheet names exactly as specified (case-sensitive).

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