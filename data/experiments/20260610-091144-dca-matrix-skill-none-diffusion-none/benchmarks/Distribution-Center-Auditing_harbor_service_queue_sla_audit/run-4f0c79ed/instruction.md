# Task Instruction

## Task: Service Queue SLA Audit

You must read `/root/Ticket_Queue.xlsx` and produce two deliverables:
1. `/root/Service_Queue_SLA_Audit.xlsx`
2. `/root/Service_Queue_SLA_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect the source workbook
- Open `/root/Ticket_Queue.xlsx` using `openpyxl` (or pandas).
- List the sheet names. Expect at least `Tickets` and `SLA_Rules`.
- Print the first few rows and all column headers of both sheets so you understand the exact column names and data types.
- Print ALL rows of `SLA_Rules` (it should be small) so you know every Priority Tier's `Max Open Hours` and `Escalation Required` values.
- Print the full `Tickets` sheet as well (or at least enough to see every unique Queue, Region, Priority Tier, and sample Escalation Code values including blanks).

#### Step 1: Load data into pandas
```python
import pandas as pd
import openpyxl
from docx import Document

# Read both sheets
tickets = pd.read_excel('/root/Ticket_Queue.xlsx', sheet_name='Tickets')
sla_rules = pd.read_excel('/root/Ticket_Queue.xlsx', sheet_name='SLA_Rules')
```
- Print `tickets.columns.tolist()` and `sla_rules.columns.tolist()` to confirm exact column names.
- Print `sla_rules` entirely.
- Print `tickets.dtypes` and check for NaN in Escalation Code column.

#### Step 2: Build the SLA lookup
- Create a dictionary from `SLA_Rules` keyed by `Priority Tier` (use the exact column name from the sheet) mapping to `Max Open Hours` and `Escalation Required`.
- Print this dictionary to verify.

#### Step 3: Build `Formatted Data`
- Start from a copy of `tickets`.
- The first 8 columns must be exactly: `Ticket ID`, `Queue`, `Priority Tier`, `Open Age Hours`, `Owner`, `Escalation Code`, `Region`, `Analyst`. If the source columns have different names, rename them. If the source has more columns, select/reorder to match exactly these 8 first.
- Compute `SLA Breach`: For each row, look up `Max Open Hours` for that row's `Priority Tier`. If `Open Age Hours > Max Open Hours`, set 1, else 0. Store as integer.
- Compute `Missing Escalation`: For each row, look up `Escalation Required` for that row's `Priority Tier`. If it is `'Y'` (or `'Yes'` — check actual values) AND the row's `Escalation Code` is blank/NaN/empty-string, set 1, else 0. Store as integer.
- Compute `Total Errors` = `SLA Breach` + `Missing Escalation` (as integer).
- Compute `Error Summary`:
  - If both are 0: `'None'`
  - If SLA Breach=1 and Missing Escalation=0: `'SLA Breach'`
  - If SLA Breach=0 and Missing Escalation=1: `'Missing Escalation'`
  - If both are 1: `'SLA Breach, Missing Escalation'`
- The final DataFrame columns (in order) must be exactly: `Ticket ID`, `Queue`, `Priority Tier`, `Open Age Hours`, `Owner`, `Escalation Code`, `Region`, `Analyst`, `SLA Breach`, `Missing Escalation`, `Total Errors`, `Error Summary`.
- Print the first 10 rows and value counts of `Error Summary` to verify.

#### Step 4: Build `Summary`
- Group `Formatted Data` by (`Queue`, `Region`).
- Aggregate: `SLA Breaches` = sum of `SLA Breach`, `Missing Escalations` = sum of `Missing Escalation`, `Total Errors` = sum of `Total Errors`.
- Filter to only groups where `Total Errors > 0`.
- Sort by `Queue` ascending then `Region` ascending.
- Rename columns to exactly: `Queue`, `Region`, `SLA Breaches`, `Missing Escalations`, `Total Errors`.
- Append a Grand Total row: `Queue`=`'Grand Total'`, `Region`=`'-'`, and the sums of the three numeric columns across the entire filtered+unfiltered dataset (i.e., the dataset-wide totals, not just the filtered rows — compute from the full Formatted Data).
- Print the Summary DataFrame to verify.

#### Step 5: Write the Excel file
```python
with pd.ExcelWriter('/root/Service_Queue_SLA_Audit.xlsx', engine='openpyxl') as writer:
    tickets.to_excel(writer, sheet_name='RawData', index=False)
    formatted.to_excel(writer, sheet_name='Formatted Data', index=False)
    summary.to_excel(writer, sheet_name='Summary', index=False)
```
- After writing, re-open the file and verify:
  - Sheet names are exactly `['RawData', 'Formatted Data', 'Summary']`.
  - `RawData` row count matches original Tickets.
  - `Formatted Data` has 12 columns with correct headers.
  - `Summary` last row has `Queue`=`'Grand Total'`.
  - Print a few sample rows from each sheet.

#### Step 6: Identify high-priority queues for the brief
- From Formatted Data, find queues with the most Total Errors. Identify at least two queue names to mention.
- Note the grand totals for SLA Breaches, Missing Escalations, Total Errors.

#### Step 7: Write the Word document
```python
doc = Document()
doc.add_heading('Service Queue SLA Audit – Executive Brief', level=1)
```
Write a paragraph (3-6 sentences) that includes ALL of these:
1. A plain-language definition of `SLA Breach` (ticket open longer than the allowed max hours for its priority tier).
2. A plain-language definition of `Missing Escalation` (ticket's priority tier requires escalation but no escalation code was recorded).
3. The exact computed totals: "X SLA Breaches, Y Missing Escalations, and Z Total Errors were identified."
4. Mention at least two specific queue names that had the highest error counts.
5. At least one actionable recommendation (e.g., implement automated escalation alerts, review staffing for the named queues, etc.).

Save as `/root/Service_Queue_SLA_Brief.docx`.

#### Step 8: Final Validation
- Verify `/root/Service_Queue_SLA_Audit.xlsx` exists and has the three correct sheet names.
- Verify `/root/Service_Queue_SLA_Brief.docx` exists.
- Re-read the Excel file and print sheet names, column headers for each sheet, row counts, and the Summary table including Grand Total row.
- Re-read the Word file and print its text to confirm all required elements are present.

### Critical Details
- Column names in output sheets must be EXACTLY as specified (case-sensitive, exact spacing).
- `SLA Breach`, `Missing Escalation`, `Total Errors` in Formatted Data must be concrete integer values (0 or 1), not formulas.
- `Error Summary` must be one of exactly these four strings: `None`, `SLA Breach`, `Missing Escalation`, `SLA Breach, Missing Escalation`.
- The Grand Total row's numeric columns must reflect dataset-wide totals (sum from all rows of Formatted Data, not just the filtered summary rows).
- Use the SLA_Rules thresholds dynamically — do NOT hardcode threshold values.
- Blank Escalation Code means NaN, empty string, or whitespace-only — treat all as blank.
- Install `python-docx` if needed: `pip install python-docx`

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