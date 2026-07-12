# Task Instruction

## Task: Service Queue SLA Audit

You must read `/root/Ticket_Queue.xlsx` and produce two deliverables:
1. `/root/Service_Queue_SLA_Audit.xlsx`
2. `/root/Service_Queue_SLA_Brief.docx`

### Step-by-step instructions

#### Step 0: Inspect the source workbook
- Open `/root/Ticket_Queue.xlsx` using `openpyxl` (or pandas).
- List the sheet names. Expect at least `Tickets` and `SLA_Rules`.
- Print the first few rows and all column headers of both sheets so you understand the exact column names, data types, and structure.
- Print all rows of `SLA_Rules` (it should be small) so you know the Priority Tier → Max Open Hours and Escalation Required mappings.
- Pay close attention to the exact column names in `Tickets` (e.g., is it `Escalation Code` or `Escalation_Code`? `Open Age Hours` or `Open_Age_Hours`?). Use the actual column names from the file.

#### Step 1: Load data into pandas
```python
import pandas as pd
from openpyxl import load_workbook

# Read source
tickets = pd.read_excel('/root/Ticket_Queue.xlsx', sheet_name='Tickets')
sla_rules = pd.read_excel('/root/Ticket_Queue.xlsx', sheet_name='SLA_Rules')

print(tickets.columns.tolist())
print(sla_rules.columns.tolist())
print(sla_rules)
print(tickets.head(10))
print(tickets.shape)
```

#### Step 2: Build the SLA lookup
- From `SLA_Rules`, create a dictionary mapping Priority Tier → {Max Open Hours, Escalation Required}.
- Use the actual column names from the sheet (print them first to be sure).

#### Step 3: Build `Formatted Data`
- Start with the tickets dataframe. The first 8 columns in the output must be exactly named:
  1. `Ticket ID`
  2. `Queue`
  3. `Priority Tier`
  4. `Open Age Hours`
  5. `Owner`
  6. `Escalation Code`
  7. `Region`
  8. `Analyst`
- If the source columns have different names (e.g., underscores, different casing), rename them to match exactly.
- Compute four new columns:
  - `SLA Breach`: For each row, look up `Max Open Hours` for that row's `Priority Tier` from the SLA rules dict. If `Open Age Hours > Max Open Hours`, set to 1, else 0.
  - `Missing Escalation`: Look up `Escalation Required` for that row's `Priority Tier`. If it is `'Y'` (or `'Yes'` — check the actual value in the data) AND the row's `Escalation Code` is blank/null/empty string, set to 1, else 0. Be careful: check for NaN, None, and empty string.
  - `Total Errors` = `SLA Breach` + `Missing Escalation`
  - `Error Summary`: Exactly one of: `'None'`, `'SLA Breach'`, `'Missing Escalation'`, `'SLA Breach, Missing Escalation'` — determined by which flags are 1.
- All four columns must contain concrete values (int for numeric, str for Error Summary), not formulas.

#### Step 4: Build `Summary`
- From `Formatted Data`, group by (`Queue`, `Region`).
- Aggregate: `SLA Breaches` = sum of `SLA Breach`, `Missing Escalations` = sum of `Missing Escalation`, `Total Errors` = sum of `Total Errors`.
- Filter to only groups where `Total Errors > 0`.
- Sort by `Queue` ascending, then `Region` ascending.
- Append a Grand Total row: Queue=`Grand Total`, Region=`-`, and the remaining columns are the dataset-wide totals (sum across ALL rows in Formatted Data, not just the filtered groups — but since groups with 0 errors contribute 0 to totals, summing the filtered groups gives the same result; however, to be safe, compute grand totals from the full Formatted Data).
- The column headers must be exactly: `Queue`, `Region`, `SLA Breaches`, `Missing Escalations`, `Total Errors`.

#### Step 5: Write the Excel output
- Write to `/root/Service_Queue_SLA_Audit.xlsx` with exactly three sheets named: `RawData`, `Formatted Data`, `Summary`.
- `RawData`: Copy the original `Tickets` data exactly (same columns, same values, same order). Use the original column names from the source.
- `Formatted Data`: The 12-column dataframe built in Step 3.
- `Summary`: The summary table built in Step 4.
- Use `pd.ExcelWriter` with `openpyxl` engine. Set `index=False` for all sheets.

#### Step 6: Verify the Excel output
- Re-read each sheet from the written file and print:
  - Sheet names
  - Column headers for each sheet
  - Row counts
  - First few rows of `Formatted Data` to confirm computed columns
  - All rows of `Summary`
  - Verify Grand Total row is present and correct
  - Verify no NaN values appear in the output (especially in `Error Summary` and `Escalation Code`)

#### Step 7: Identify top queues for the Word brief
- From the Summary (excluding Grand Total), find the queues with the highest Total Errors. Note at least two queue names.
- Note the grand totals for SLA Breaches, Missing Escalations, Total Errors.

#### Step 8: Write the Word document
- Use `python-docx` to create `/root/Service_Queue_SLA_Brief.docx`.
- Write an executive summary paragraph (3-6 sentences) that includes:
  1. A plain-language definition of both checks: SLA Breach means a ticket's open age exceeds the maximum allowed hours for its priority tier; Missing Escalation means a ticket's priority tier requires escalation but no escalation code was recorded.
  2. The computed totals: "The audit identified X SLA Breaches, Y Missing Escalations, and Z Total Errors across all queues."
  3. Mention at least two specific high-priority queues by name with their error counts.
  4. At least one actionable recommendation (e.g., "We recommend immediate triage of breached tickets in [Queue A] and [Queue B], and implementing automated escalation assignment to reduce missing escalation codes.").
- Save the file.

#### Step 9: Final verification
- Confirm both files exist: `/root/Service_Queue_SLA_Audit.xlsx` and `/root/Service_Queue_SLA_Brief.docx`.
- Re-read the Word doc and print its text to confirm content.
- Re-read the Excel file one more time and print summary stats to confirm correctness.

### Critical details to watch for:
- Column name matching: The source file may use underscores or different casing. Always use the actual column names from the file, and rename to the exact required names for the output.
- Blank escalation codes: Check for `NaN`, `None`, empty string `''`, and whitespace-only strings when determining `Missing Escalation`.
- The `Escalation Required` field in SLA_Rules might say `Y`/`N` or `Yes`/`No` — check the actual values.
- All numeric flag columns (SLA Breach, Missing Escalation, Total Errors) should be Python int, not float.
- Error Summary must use exact strings with exact punctuation including the comma-space in `'SLA Breach, Missing Escalation'`.
- Sheet names must be exactly `RawData`, `Formatted Data`, `Summary` (note the space in `Formatted Data`).
- Install `python-docx` if needed: `pip install python-docx`.
- The Summary Grand Total row must have Queue=`Grand Total` and Region=`-` (a single hyphen).

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