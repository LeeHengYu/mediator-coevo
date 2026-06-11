# Task Instruction

## Task: Build Cycle Count Variance Audit Files

You must create two output files:
1. `/root/Cycle_Count_Variance_Audit.xlsx`
2. `/root/Cycle_Count_Variance_Brief.docx`

### Step 0: Inspect Input Files

Before writing any code, inspect the three input files to understand their structure:

```bash
cd /root
python3 -c "
import openpyxl
for fname in ['Cycle_Plan.xlsx', 'Count_Event_Log.xlsx', 'Cycle_Template.xlsx']:
    wb = openpyxl.load_workbook(fname)
    print(f'\n=== {fname} ===')
    print(f'Sheets: {wb.sheetnames}')
    for sn in wb.sheetnames:
        ws = wb[sn]
        print(f'  Sheet: {sn}, rows={ws.max_row}, cols={ws.max_column}')
        for r in range(1, min(ws.max_row+1, 8)):
            row_data = [ws.cell(r, c).value for c in range(1, ws.max_column+1)]
            print(f'    Row {r}: {row_data}')
"
```

Note the exact column names, sheet names, and data formats. Pay special attention to:
- The column headers in Cycle_Plan.xlsx (these become RawData and the first 7 columns of Formatted Data)
- The column headers and values in Count_Event_Log.xlsx (especially the Event Type column and Count Qty column)
- The Overview sheet structure in Cycle_Template.xlsx

### Step 1: Write the Python Script

Create `/root/build_audit.py` with the following logic. Use `openpyxl` for Excel and `python-docx` for Word.

First, install dependencies if needed:
```bash
pip install openpyxl python-docx
```

Then write the script. The script must:

#### A) Load Data
- Load `Cycle_Plan.xlsx` — read the plan table (likely the first/only data sheet). Store all rows preserving order.
- Load `Count_Event_Log.xlsx` — read all event rows.
- Load `Cycle_Template.xlsx` — specifically the `Overview` sheet.

#### B) Process Count Event Log
- Filter to rows where `Event Type` equals `FINAL` (case-insensitive comparison).
- Exclude rows where any of `Facility`, `Session ID`, `Bin ID`, or `Count Qty` is blank/None.
- For each unique `(Facility, Session ID, Bin ID)` key, keep only the LAST row (by row order in the spreadsheet, i.e., the latest entry).
- Build a lookup dictionary: key = `(Facility, Session ID, Bin ID)` → value = `Count Qty` (as a number).

#### C) Build the Output Workbook

Create a new workbook with exactly 4 sheets in this order: `Overview`, `RawData`, `Formatted Data`, `Summary`.

**Overview sheet:** Copy from `Cycle_Template.xlsx`'s `Overview` sheet cell-by-cell, preserving all values exactly. Copy merged cells if present. Do NOT modify any content.

**RawData sheet:** Copy the plan table from `Cycle_Plan.xlsx` exactly — headers and all data rows, preserving order and values.

**Formatted Data sheet:**
- First 7 columns: same as RawData (Facility, Session ID, Bin ID, Product ID, Expected Qty, Allowed Variance, Approval Needed) — use the EXACT header names from the plan file.
- Column 8: `Missing Final Count`
- Column 9: `Approval Gap`
- Column 10: `Total Errors`
- Column 11: `Error Summary`

For each row:
- Look up `(Facility, Session ID, Bin ID)` in the FINAL event dictionary.
- `Missing Final Count` = 1 if key not found, else 0.
- `Approval Gap` = 1 if ALL of: (a) key IS found, (b) `Approval Needed` equals `YES` case-insensitively, (c) `abs(Expected Qty - Count Qty) > Allowed Variance`. Otherwise 0.
- `Total Errors` = `Missing Final Count + Approval Gap` (write as integer).
- `Error Summary`: construct from the active flags:
  - If both flags are 0: `None`
  - If only Missing Final Count=1: `Missing Final Count`
  - If only Approval Gap=1: `Approval Gap`
  - If both=1: `Missing Final Count, Approval Gap`

Write all values as concrete numbers/strings (no Excel formulas).

**Summary sheet:**
- Headers: `Facility`, `Session ID`, `Missing Final Counts`, `Approval Gaps`, `Total Errors`
- Aggregate from Formatted Data by `(Facility, Session ID)`.
- Sum `Missing Final Count` → `Missing Final Counts`, sum `Approval Gap` → `Approval Gaps`, sum `Total Errors` → `Total Errors` for each group.
- Include ONLY groups where `Total Errors > 0`.
- Sort by `Facility` ascending, then `Session ID` ascending.
- Append a Grand Total row: `Facility`=`Grand Total`, `Session ID`=`-`, and sums of the three numeric columns across all included groups.

Save as `/root/Cycle_Count_Variance_Audit.xlsx`.

#### D) Build the Word Document

This is CRITICAL — previous execution failed here. The test `test_word_summary_content` searches for facility-session combinations in the text.

From the Summary data, identify the top 2 (Facility, Session ID) pairs with the highest `Total Errors` (break ties by Facility then Session ID ascending).

Compute grand totals: total Missing Final Counts, total Approval Gaps, total Total Errors.

Write a 3-6 sentence executive summary paragraph that includes ALL of the following:

1. **Plain-language definition of both checks:**
   - "Missing Final Count" flags bins where no final count event was recorded.
   - "Approval Gap" flags bins where the counted quantity deviates from the expected quantity beyond the allowed variance threshold and requires approval.

2. **Computed totals:** State the exact numbers, e.g., "The audit identified X Missing Final Counts, Y Approval Gaps, and Z Total Errors across all sessions."

3. **At least one actionable recommendation**, e.g., "We recommend prioritizing recounts for bins with missing final counts and escalating approval gap cases to supervisors."

4. **CRITICAL: Mention at least two high-priority facility-session combinations.** Format them EXACTLY as `{Facility}-{Session ID}` (hyphen-separated, no spaces around the hyphen). For example, if Facility is `FAC01` and Session ID is `SESS05`, write `FAC01-SESS05`. Include these directly in the text, e.g.:
   "The highest-priority sessions requiring immediate attention are FAC01-SESS05 and FAC02-SESS12, which together account for the majority of exceptions."

   The test uses a regex or string search for patterns like `FACILITY-SESSION` or `FACILITY SESSION`. Using the hyphen-separated format ensures detection.

Save as `/root/Cycle_Count_Variance_Brief.docx`.

### Step 2: Run the Script

```bash
cd /root && python3 build_audit.py
```

Check for any errors. If errors occur, fix and re-run.

### Step 3: Validate Output

Run validation checks:

```python
import openpyxl

# Check sheet names
wb = openpyxl.load_workbook('/root/Cycle_Count_Variance_Audit.xlsx')
print('Sheets:', wb.sheetnames)
assert wb.sheetnames == ['Overview', 'RawData', 'Formatted Data', 'Summary']

# Check Formatted Data headers
ws = wb['Formatted Data']
headers = [ws.cell(1, c).value for c in range(1, 12)]
print('Formatted Data headers:', headers)
assert headers[7] == 'Missing Final Count'
assert headers[8] == 'Approval Gap'
assert headers[9] == 'Total Errors'
assert headers[10] == 'Error Summary'

# Check Summary headers
ws2 = wb['Summary']
sum_headers = [ws2.cell(1, c).value for c in range(1, 6)]
print('Summary headers:', sum_headers)
assert sum_headers == ['Facility', 'Session ID', 'Missing Final Counts', 'Approval Gaps', 'Total Errors']

# Check last row is Grand Total
last_row = ws2.max_row
print(f'Summary last row ({last_row}):', [ws2.cell(last_row, c).value for c in range(1, 6)])
assert ws2.cell(last_row, 1).value == 'Grand Total'
assert ws2.cell(last_row, 2).value == '-'

print('\nAll structural checks passed!')
```

Also validate the Word document:
```python
from docx import Document
doc = Document('/root/Cycle_Count_Variance_Brief.docx')
text = ' '.join([p.text for p in doc.paragraphs]).lower()
print('Word text:', text[:500])
assert 'missing final count' in text, 'Missing Final Count not mentioned'
assert 'approval gap' in text, 'Approval Gap not mentioned'
# Check for facility-session patterns
import re
matches = re.findall(r'[A-Za-z0-9_]+-[A-Za-z0-9_]+', ' '.join([p.text for p in doc.paragraphs]))
print('Potential facility-session matches:', matches)
assert len([m for m in matches if len(m) > 3]) >= 2, 'Need at least 2 facility-session combos'
print('Word doc checks passed!')
```

### Step 4: Run the test suite if available

```bash
if [ -f /root/tests/test_outputs.py ]; then
    cd /root && python3 -m pytest tests/test_outputs.py -v
elif [ -f /root/test_outputs.py ]; then
    cd /root && python3 -m pytest test_outputs.py -v
fi
```

If tests fail, read the error messages carefully, fix the script, and re-run. Pay particular attention to:
- The Word document content matching (facility-session format)
- Exact header names
- Data type correctness (integers vs strings)
- The Overview sheet being truly unchanged

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