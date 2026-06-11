# Task Instruction

Execute the following steps to produce `/root/Service_Queue_SLA_Audit.xlsx` and `/root/Service_Queue_SLA_Brief.docx`.

## Step 1: Inspect the source workbook

```python
import pandas as pd

# Read both sheets
tickets = pd.read_excel('/root/Ticket_Queue.xlsx', sheet_name='Tickets')
sla_rules = pd.read_excel('/root/Ticket_Queue.xlsx', sheet_name='SLA_Rules')

print('Tickets columns:', tickets.columns.tolist())
print('Tickets shape:', tickets.shape)
print('Tickets head:')
print(tickets.head(10).to_string())
print()
print('SLA_Rules columns:', sla_rules.columns.tolist())
print('SLA_Rules shape:', sla_rules.shape)
print('SLA_Rules:')
print(sla_rules.to_string())
```

Carefully note the exact column names in both sheets. Identify which column maps to each required output header.

## Step 2: Build the output Excel workbook

After inspecting column names, write a single Python script that:

### 2a) RawData sheet
- Copy the `Tickets` DataFrame exactly as-is (same columns, same values, same order).
- **Critical**: Do NOT convert NaN to None or drop any values. Write the DataFrame directly with `to_excel` so that empty cells in the source remain as empty cells in the output. If the source file contains literal strings like 'N/A' in any cell, those must be preserved exactly. Read the source carefully to determine whether blank cells contain NaN or a string like 'N/A'.

### 2b) Formatted Data sheet
- Start with the same rows in the same order.
- Rename/select columns to produce exactly these 8 columns in order:
  1. `Ticket ID`
  2. `Queue`
  3. `Priority Tier`
  4. `Open Age Hours`
  5. `Owner`
  6. `Escalation Code`
  7. `Region`
  8. `Analyst`
- Map source column names to these output names. For example, if the source has `Ticket_ID` or `TicketID`, map it to `Ticket ID`. Inspect the actual column names from Step 1 to determine the mapping.
- Build a lookup dict from `SLA_Rules`: for each Priority Tier, store `Max Open Hours` and `Escalation Required`.
- Compute four new columns (as concrete values, not formulas):
  - `SLA Breach`: 1 if `Open Age Hours` > `Max Open Hours` for that row's Priority Tier, else 0. Use int type.
  - `Missing Escalation`: 1 if the SLA rule says `Escalation Required` == 'Y' for that Priority Tier AND the row's `Escalation Code` is blank (NaN, None, empty string, or whitespace-only), else 0. Use int type.
  - `Total Errors`: `SLA Breach` + `Missing Escalation` (int).
  - `Error Summary`: exactly one of: `'None'`, `'SLA Breach'`, `'Missing Escalation'`, `'SLA Breach, Missing Escalation'` — determined by which flags are 1.

### 2c) Summary sheet
- Group `Formatted Data` by `(Queue, Region)`.
- Sum `SLA Breach` → `SLA Breaches`, `Missing Escalation` → `Missing Escalations`, `Total Errors` → `Total Errors`.
- Filter to only groups where `Total Errors > 0`.
- Sort by `Queue` ascending then `Region` ascending.
- Append a Grand Total row: Queue=`Grand Total`, Region=`-`, and sums of the three numeric columns across all rows of Formatted Data (not just the filtered groups — use the full dataset totals).
- Headers must be exactly: `Queue`, `Region`, `SLA Breaches`, `Missing Escalations`, `Total Errors`.

### 2d) Write the workbook
- Use `openpyxl` engine with `pd.ExcelWriter`.
- Sheet names must be exactly: `RawData`, `Formatted Data`, `Summary`.
- Write without the pandas index (`index=False`).

## Step 3: Build the Word document

Using `python-docx`:
- Create `/root/Service_Queue_SLA_Brief.docx`.
- Write an executive summary paragraph (3-6 sentences) that includes:
  - Plain-language definition of both checks: SLA Breach (ticket open longer than the allowed max hours for its priority tier) and Missing Escalation (ticket's priority tier requires escalation but no escalation code is recorded).
  - The computed totals for SLA Breaches, Missing Escalations, and Total Errors (use the actual numbers from the data).
  - At least one actionable recommendation (e.g., implement automated escalation alerts).
  - Mention at least two queues that have the most errors by name.

## Step 4: Validate

After creating both files, run validation:
```python
import pandas as pd

# Validate Excel
xl = pd.ExcelFile('/root/Service_Queue_SLA_Audit.xlsx')
print('Sheet names:', xl.sheet_names)

raw = pd.read_excel(xl, 'RawData')
print('RawData shape:', raw.shape)
print('RawData columns:', raw.columns.tolist())
print(raw.head())

fmt = pd.read_excel(xl, 'Formatted Data')
print('Formatted Data shape:', fmt.shape)
print('Formatted Data columns:', fmt.columns.tolist())
print(fmt.head(10).to_string())
# Verify column count is 12
assert len(fmt.columns) == 12, f'Expected 12 columns, got {len(fmt.columns)}'
# Verify computed columns are integers
assert fmt['SLA Breach'].dtype in ['int64', 'int32', 'float64']
assert fmt['Total Errors'].dtype in ['int64', 'int32', 'float64']

summ = pd.read_excel(xl, 'Summary')
print('Summary shape:', summ.shape)
print('Summary columns:', summ.columns.tolist())
print(summ.to_string())
# Verify last row is Grand Total
assert summ.iloc[-1]['Queue'] == 'Grand Total'
assert summ.iloc[-1]['Region'] == '-'

# Validate Word
from docx import Document
doc = Document('/root/Service_Queue_SLA_Brief.docx')
text = ' '.join([p.text for p in doc.paragraphs])
print('Word doc text length:', len(text))
print('Word doc text:', text[:500])
assert 'SLA Breach' in text or 'SLA breach' in text or 'SLA breaches' in text
assert 'Missing Escalation' in text or 'missing escalation' in text
print('All validations passed.')
```

## Important Notes
- Install any needed packages: `pip install openpyxl python-docx` if not already available.
- When reading the source Excel, pay close attention to how blank/empty cells appear. If they show as NaN in pandas, and the RawData sheet should be an exact copy, write them back as-is (they will appear as empty cells in Excel). Do NOT fill NaN with 'N/A' or any other string unless the source data itself contains that string.
- For the `Missing Escalation` check, treat NaN, None, empty string '', and whitespace-only strings as "blank" escalation codes.
- Use `int` type for the flag columns (SLA Breach, Missing Escalation, Total Errors) to avoid writing 1.0/0.0 instead of 1/0.
- The Summary numeric columns should also be integers.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=hard, tags=[excel, openpyxl, docx, audit, service].
Verifier config: timeout_sec=900.0.