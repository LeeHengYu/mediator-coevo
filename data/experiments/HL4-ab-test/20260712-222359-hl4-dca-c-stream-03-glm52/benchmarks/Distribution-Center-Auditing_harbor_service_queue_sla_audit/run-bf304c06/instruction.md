# Task Instruction

## Task: Service Queue SLA Audit

You must read `/root/Ticket_Queue.xlsx` and produce two deliverables:
1. `/root/Service_Queue_SLA_Audit.xlsx`
2. `/root/Service_Queue_SLA_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect the source workbook
- Open `/root/Ticket_Queue.xlsx` using `openpyxl` (or pandas).
- List all sheet names. Expect at least `Tickets` and `SLA_Rules`.
- Print the first few rows and all column headers of both sheets so you understand the exact column names and data types.
- Print the full `SLA_Rules` table (it should be small) so you know every Priority Tier's `Max Open Hours` and `Escalation Required` values.
- Pay close attention to the exact column names in `Tickets` — they must map to the 8 required columns: `Ticket ID`, `Queue`, `Priority Tier`, `Open Age Hours`, `Owner`, `Escalation Code`, `Region`, `Analyst`. If the source uses slightly different names, note the mapping.

#### Step 1: Load data into pandas
```python
import pandas as pd
import numpy as np

tickets = pd.read_excel('/root/Ticket_Queue.xlsx', sheet_name='Tickets')
sla_rules = pd.read_excel('/root/Ticket_Queue.xlsx', sheet_name='SLA_Rules')
```
- Print `tickets.columns.tolist()` and `sla_rules.columns.tolist()` to confirm exact names.
- Print `sla_rules` in full.
- Print `tickets.shape` and `tickets.head(10)`.

#### Step 2: Build the RawData sheet content
- `raw_data = tickets.copy()` — keep every column and row exactly as-is.

#### Step 3: Build the Formatted Data sheet
- Start from `tickets.copy()`.
- Merge/map SLA rules onto tickets by `Priority Tier`.
- Compute the four new columns as concrete values (integers and strings, NOT formulas):
  - `SLA Breach`: 1 if `Open Age Hours` > the matched `Max Open Hours`, else 0. Use int type.
  - `Missing Escalation`: 1 if the SLA rule says `Escalation Required` == 'Y' (for that priority tier) AND the ticket's `Escalation Code` is blank/null/empty string, else 0. Use int type.
  - `Total Errors`: `SLA Breach` + `Missing Escalation` (int).
  - `Error Summary`: Exactly one of these four strings:
    - `'None'` (when both are 0)
    - `'SLA Breach'` (when only SLA Breach is 1)
    - `'Missing Escalation'` (when only Missing Escalation is 1)
    - `'SLA Breach, Missing Escalation'` (when both are 1)
- Keep only these 12 columns in this exact order with these exact headers:
  1. `Ticket ID`
  2. `Queue`
  3. `Priority Tier`
  4. `Open Age Hours`
  5. `Owner`
  6. `Escalation Code`
  7. `Region`
  8. `Analyst`
  9. `SLA Breach`
  10. `Missing Escalation`
  11. `Total Errors`
  12. `Error Summary`
- If the source column names differ from these target names, rename them.
- Preserve the original row order (same as RawData).

#### Step 4: Build the Summary sheet
- From the Formatted Data DataFrame, group by `['Queue', 'Region']`.
- Aggregate: `SLA Breaches` = sum of `SLA Breach`, `Missing Escalations` = sum of `Missing Escalation`, `Total Errors` = sum of `Total Errors`.
- Filter to only groups where `Total Errors > 0`.
- Sort by `Queue` ascending, then `Region` ascending.
- Append a Grand Total row: `Queue` = `'Grand Total'`, `Region` = `'-'`, and the remaining three columns = dataset-wide totals (sum over ALL rows in Formatted Data, not just filtered groups — but since filtered groups excluded zero-error groups, the totals should be the same; still, compute from the full Formatted Data to be safe).
- The final column headers must be exactly: `Queue`, `Region`, `SLA Breaches`, `Missing Escalations`, `Total Errors`.

#### Step 5: Write the Excel file
```python
with pd.ExcelWriter('/root/Service_Queue_SLA_Audit.xlsx', engine='openpyxl') as writer:
    raw_data.to_excel(writer, sheet_name='RawData', index=False)
    formatted_data.to_excel(writer, sheet_name='Formatted Data', index=False)
    summary.to_excel(writer, sheet_name='Summary', index=False)
```
- Verify the file was created and has exactly three sheets named `RawData`, `Formatted Data`, `Summary`.
- Re-read and print the shape and first few rows of each sheet to confirm correctness.

#### Step 6: Write the Word document
```python
from docx import Document
```
- Create `/root/Service_Queue_SLA_Brief.docx` with a short executive summary (3–6 sentences) that includes:
  1. A plain-language definition of both checks: explain that an SLA Breach occurs when a ticket's open age exceeds the maximum allowed hours for its priority tier, and a Missing Escalation occurs when a ticket's priority tier requires escalation but no escalation code is recorded.
  2. The computed totals: state the exact numbers for total SLA Breaches, total Missing Escalations, and total Total Errors from the Grand Total row.
  3. At least one actionable recommendation (e.g., implement automated escalation alerts, review staffing for high-breach queues).
  4. Mention at least two specific queues that have the highest error counts (look at the Summary data to identify them — pick the top 2 queues by Total Errors).
- Save the document.

#### Step 7: Final Validation
- Re-open `/root/Service_Queue_SLA_Audit.xlsx` and verify:
  - Sheet names are exactly `['RawData', 'Formatted Data', 'Summary']`.
  - `RawData` row count matches source `Tickets` row count.
  - `Formatted Data` has 12 columns with exact expected headers.
  - `Formatted Data` row count matches `RawData`.
  - `Summary` has 5 columns with exact expected headers.
  - `Summary` last row has Queue = 'Grand Total' and Region = '-'.
  - Grand Total numbers match the sums from Formatted Data.
- Re-open `/root/Service_Queue_SLA_Brief.docx` and print its text to confirm it contains the required elements.
- Print 'ALL VALIDATIONS PASSED' if everything checks out.

### Critical Reminders
- Do NOT hardcode SLA thresholds — always read them from `SLA_Rules`.
- Escalation Code being blank means it's NaN/None/empty string — check with `pd.isna()` or equivalent, and also check for empty strings after stripping.
- Write concrete values (not Excel formulas) for columns 9-12.
- Exact file paths and sheet names matter for grading.

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