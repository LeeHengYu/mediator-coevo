# Task Instruction

Execute the following steps in a single Python script to produce `/root/Service_Queue_SLA_Audit.xlsx` and `/root/Service_Queue_SLA_Brief.docx`.

## Step 0 – Inspect the source workbook
```python
import openpyxl
wb = openpyxl.load_workbook('/root/Ticket_Queue.xlsx')
print('Sheet names:', wb.sheetnames)
for s in wb.sheetnames:
    ws = wb[s]
    print(f'\n--- {s} (rows={ws.max_row}, cols={ws.max_column}) ---')
    for r in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 10), values_only=True):
        print(r)
```
Print the output so you can see exact column names, data types, and SLA_Rules content before writing any logic.

## Step 1 – Read data into Pandas
```python
import pandas as pd

tickets = pd.read_excel('/root/Ticket_Queue.xlsx', sheet_name='Tickets')
sla_rules = pd.read_excel('/root/Ticket_Queue.xlsx', sheet_name='SLA_Rules')

print(tickets.dtypes)
print(tickets.head())
print(sla_rules)
```
Confirm column names exactly. If they differ from the expected names below, adjust all downstream references accordingly.

## Step 2 – Build RawData
RawData is an exact copy of the Tickets sheet (same columns, same order).

## Step 3 – Build Formatted Data
1. Start from a copy of tickets.
2. Merge (left join) with `sla_rules` on `Priority Tier` to bring in `Max Open Hours` and `Escalation Required`.
   - **Type safety**: convert `Open Age Hours` and `Max Open Hours` to float before comparison.
   - **String safety**: strip whitespace from `Priority Tier` in both frames before merging.
3. Compute columns:
   - `SLA Breach` = 1 where `Open Age Hours > Max Open Hours`, else 0.
   - `Missing Escalation` = 1 where `Escalation Required` is 'Y' (case-insensitive, stripped) AND `Escalation Code` is blank/NaN/empty-string, else 0.
     - To detect blank: `escalation_code.isna() | (escalation_code.astype(str).str.strip() == '') | (escalation_code.astype(str).str.upper() == 'NAN')`.
   - `Total Errors` = `SLA Breach` + `Missing Escalation`.
   - `Error Summary`:
     - Both flags 1 → `'SLA Breach, Missing Escalation'`
     - Only SLA Breach → `'SLA Breach'`
     - Only Missing Escalation → `'Missing Escalation'`
     - Neither → `'None'`
4. Keep only these 12 columns in this exact order:
   `Ticket ID, Queue, Priority Tier, Open Age Hours, Owner, Escalation Code, Region, Analyst, SLA Breach, Missing Escalation, Total Errors, Error Summary`
5. Ensure all four new columns are concrete values (int for the numeric ones, str for Error Summary).

## Step 4 – Build Summary
1. From the Formatted Data frame, group by `(Queue, Region)` and sum `SLA Breach` (→ `SLA Breaches`), `Missing Escalation` (→ `Missing Escalations`), `Total Errors`.
2. Filter to groups where `Total Errors > 0`.
3. Sort ascending by `Queue` then `Region`.
4. Append a Grand Total row: `Queue='Grand Total'`, `Region='-'`, and sums across the **full Formatted Data** (not just filtered groups).
5. Final columns: `Queue, Region, SLA Breaches, Missing Escalations, Total Errors`.

## Step 5 – Write Excel
Use `pd.ExcelWriter('/root/Service_Queue_SLA_Audit.xlsx', engine='openpyxl')` and write:
- `raw_data_df` → sheet `RawData` (index=False)
- `formatted_df` → sheet `Formatted Data` (index=False)
- `summary_df` → sheet `Summary` (index=False)

After writing, re-read the file and print sheet names, column headers, row counts, and a few sample rows from each sheet to verify correctness.

## Step 6 – Identify top queues for the Word brief
From the Summary (excluding Grand Total), find the two queues with the highest `Total Errors`. Store their names.

## Step 7 – Write Word document
Using `python-docx`:
1. Add a heading "Service Queue SLA Audit – Executive Summary".
2. Write 3-6 sentences that include:
   - Definition of SLA Breach: "A ticket is flagged as an SLA Breach when its Open Age Hours exceeds the maximum allowed hours for its Priority Tier as defined in SLA_Rules."
   - Definition of Missing Escalation: "A ticket is flagged as a Missing Escalation when its Priority Tier requires escalation per SLA_Rules but no Escalation Code is recorded."
   - Exact totals: "Across the dataset, there were {X} SLA Breaches, {Y} Missing Escalations, and {Z} Total Errors."
   - Mention the two high-priority queues by name: "The queues with the most frequent exceptions were {Q1} and {Q2}."
   - Actionable recommendation: "We recommend prioritizing SLA compliance reviews for these queues and implementing automated escalation alerts to reduce missing escalation incidents."
3. Save to `/root/Service_Queue_SLA_Brief.docx`.

## Step 8 – Final Validation
Re-read both output files and print:
- Excel: sheet names, each sheet's columns and row count, first 3 and last 3 rows of Formatted Data and Summary.
- Word: full text content.
Confirm everything matches the spec before finishing.

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