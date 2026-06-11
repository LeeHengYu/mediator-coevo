# Task Instruction

## Task: Service Queue SLA Audit

You must read `/root/Ticket_Queue.xlsx` and produce two deliverables:
1. `/root/Service_Queue_SLA_Audit.xlsx`
2. `/root/Service_Queue_SLA_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect the source workbook
- Open `/root/Ticket_Queue.xlsx` using `openpyxl` (or pandas).
- List the sheet names. Expect at least `Tickets` and `SLA_Rules`.
- Print the first few rows AND all column headers of each sheet so you understand the exact column names and data types.
- Print all rows of `SLA_Rules` (it should be small) so you know every Priority Tier's `Max Open Hours` and `Escalation Required` values.
- Pay special attention to: Are column names exactly `Priority Tier`, `Max Open Hours`, `Escalation Required`? Or are they slightly different? Use the actual names.

#### Step 1: Load data into pandas
```python
import pandas as pd
from openpyxl import load_workbook

tickets = pd.read_excel('/root/Ticket_Queue.xlsx', sheet_name='Tickets')
sla_rules = pd.read_excel('/root/Ticket_Queue.xlsx', sheet_name='SLA_Rules')

print('Tickets columns:', list(tickets.columns))
print('Tickets shape:', tickets.shape)
print(tickets.head(10))
print('SLA_Rules columns:', list(sla_rules.columns))
print(sla_rules)
```

#### Step 2: Build the SLA lookup
- Create a dictionary from `SLA_Rules` mapping each Priority Tier to its `Max Open Hours` and `Escalation Required` values.
- Print this dictionary to verify.

#### Step 3: Create `RawData` sheet content
- This is just an exact copy of the `Tickets` data. Keep it as-is.

#### Step 4: Create `Formatted Data` sheet content
- Start with the Tickets data.
- The first 8 columns must be exactly: `Ticket ID`, `Queue`, `Priority Tier`, `Open Age Hours`, `Owner`, `Escalation Code`, `Region`, `Analyst`. Select/rename columns from the source to match these names exactly. Verify the source has corresponding columns (they might already match or be slightly different).
- For each row, look up the SLA rule for that row's `Priority Tier`:
  - `SLA Breach`: 1 if `Open Age Hours` > `Max Open Hours` for that tier, else 0
  - `Missing Escalation`: 1 if `Escalation Required` == 'Y' for that tier AND `Escalation Code` is blank/NaN/empty, else 0
  - `Total Errors`: `SLA Breach` + `Missing Escalation`
  - `Error Summary`: Exactly one of: `None`, `SLA Breach`, `Missing Escalation`, `SLA Breach, Missing Escalation` (note the comma-space separator)
- Write these as concrete integer/string values (not formulas).
- Keep the same row order as RawData.

#### Step 5: Create `Summary` sheet content
- Group `Formatted Data` by (`Queue`, `Region`).
- Aggregate: `SLA Breaches` = sum of `SLA Breach`, `Missing Escalations` = sum of `Missing Escalation`, `Total Errors` = sum of `Total Errors`.
- Filter to only groups where `Total Errors > 0`.
- Sort by `Queue` ascending, then `Region` ascending.
- Append a Grand Total row: Queue=`Grand Total`, Region=`-`, and the remaining columns are the dataset-wide totals (sum across ALL rows in Formatted Data, not just the filtered groups — actually, since groups with 0 errors are excluded from the table but the grand total should reflect the entire dataset totals, let me clarify: the grand total should be the sum of SLA Breaches, Missing Escalations, and Total Errors across ALL tickets in Formatted Data).
- Headers must be exactly: `Queue`, `Region`, `SLA Breaches`, `Missing Escalations`, `Total Errors`.

#### Step 6: Write the Excel file
- Use `openpyxl` or `pd.ExcelWriter` to write `/root/Service_Queue_SLA_Audit.xlsx` with exactly three sheets named `RawData`, `Formatted Data`, `Summary` (in that order if possible).
- Ensure integer columns are written as integers (not floats like 1.0).
- Verify the file by re-reading it and printing sheet names, row counts, and a sample of each sheet.

#### Step 7: Write the Word document
- Use `python-docx` to create `/root/Service_Queue_SLA_Brief.docx`.
- Write an executive summary paragraph (3-6 sentences) that includes:
  - Plain-language definition of both checks: SLA Breach (ticket open longer than the allowed max hours for its priority tier) and Missing Escalation (ticket's priority tier requires escalation but no escalation code is recorded).
  - The computed totals: total SLA Breaches, total Missing Escalations, total Total Errors (use the actual numbers from your computation).
  - At least one actionable recommendation (e.g., "We recommend immediate triage of breached high-priority tickets and mandatory escalation code entry for tiers requiring escalation.").
  - Mention at least two specific queues that have the highest error counts (look at the Summary data to identify them).

#### Step 8: Final Validation
- Re-read `/root/Service_Queue_SLA_Audit.xlsx` and verify:
  - Sheet names are exactly `RawData`, `Formatted Data`, `Summary`
  - `RawData` row count matches source `Tickets` row count
  - `Formatted Data` has 12 columns with correct headers
  - `Formatted Data` row count matches `RawData`
  - `Summary` last row has Queue = `Grand Total`
  - Summary Grand Total numbers match the sums from Formatted Data
  - All `Error Summary` values are one of the four allowed strings
  - `SLA Breach` and `Missing Escalation` columns contain only 0 or 1
- Verify `/root/Service_Queue_SLA_Brief.docx` exists and contains text.
- Print confirmation of all checks passing.

### Important Notes
- Do NOT hardcode SLA thresholds. Read them from `SLA_Rules`.
- Filenames and sheet names must be EXACTLY as specified (case-sensitive).
- Install any needed packages (`pip install openpyxl python-docx`) if not already available.
- If `Escalation Code` column has NaN, empty string, or None, treat all as blank for the Missing Escalation check.
- The `Error Summary` string `None` is the literal text string "None", not Python's None object.

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