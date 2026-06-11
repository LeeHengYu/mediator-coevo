# Task Instruction

Execute the following steps in a single Python script to produce the two deliverables.

## Step 0 – Install dependencies
```
pip install openpyxl python-docx
```

## Step 1 – Read source workbook
- Open `/root/Ticket_Queue.xlsx` with openpyxl (data_only=True).
- Read the `Tickets` sheet: capture the header row and all data rows. Print the headers so you can see the exact column names.
- Read the `SLA_Rules` sheet: capture the header row and all data rows. Print the headers and all rows.

## Step 2 – Build lookup from SLA_Rules
- From `SLA_Rules`, build a dictionary keyed by `Priority Tier` (use the exact column name found in the sheet – it may be "Priority Tier", "Priority", etc.). Each value should store `Max Open Hours` (numeric) and `Escalation Required` (string, e.g. "Y" or "N"). Print the resulting dictionary for verification.

## Step 3 – Map Tickets columns
Identify the column indices (0-based) in the Tickets header for:
1. Ticket ID
2. Queue
3. Priority Tier
4. Open Age Hours
5. Owner
6. Escalation Code
7. Region
8. Analyst

Use flexible matching: strip whitespace, try common synonyms if needed. Print the mapping.

## Step 4 – Create output workbook `/root/Service_Queue_SLA_Audit.xlsx`

### Sheet `RawData`
- Copy the Tickets header row and all data rows exactly as-is.

### Sheet `Formatted Data`
- Header row: the 8 column names listed above, then: `SLA Breach`, `Missing Escalation`, `Total Errors`, `Error Summary`.
- For each ticket row (same order as RawData):
  - Extract the 8 base values using the column mapping.
  - Look up the ticket's Priority Tier in the SLA_Rules dict.
  - `SLA Breach` = 1 if `Open Age Hours` > `Max Open Hours`, else 0. Treat None/blank Open Age Hours as 0.
  - `Missing Escalation` = 1 if `Escalation Required` == "Y" AND (`Escalation Code` is None or blank string), else 0.
  - `Total Errors` = SLA Breach + Missing Escalation.
  - `Error Summary`:
    - If Total Errors == 0 → "None"
    - If SLA Breach == 1 and Missing Escalation == 0 → "SLA Breach"
    - If SLA Breach == 0 and Missing Escalation == 1 → "Missing Escalation"
    - If both == 1 → "SLA Breach, Missing Escalation"
  - Write concrete integers and strings (no formulas).

### Sheet `Summary`
- Aggregate from the Formatted Data rows by (Queue, Region).
- For each group, sum SLA Breaches, Missing Escalations, Total Errors.
- Keep only groups where Total Errors > 0.
- Sort by Queue ascending, then Region ascending (case-sensitive default sort is fine).
- Headers: `Queue`, `Region`, `SLA Breaches`, `Missing Escalations`, `Total Errors`.
- Append a final row: `Grand Total`, `-`, and the dataset-wide sums of the three numeric columns.

Save the workbook.

## Step 5 – Compute stats for the Word brief
- Compute grand totals: total SLA Breaches, total Missing Escalations, total Total Errors.
- Identify the top 2 (Queue) names by Total Errors descending (aggregate across all regions). These are the "high-priority queues" to mention.

## Step 6 – Create `/root/Service_Queue_SLA_Brief.docx`
Using python-docx, create a document with a single paragraph (or a few short paragraphs, 3-6 sentences total) that:
- Defines SLA Breach: a ticket whose Open Age Hours exceeds the maximum allowed for its Priority Tier.
- Defines Missing Escalation: a ticket whose Priority Tier requires escalation but has no Escalation Code recorded.
- States the computed totals: X SLA Breaches, Y Missing Escalations, Z Total Errors.
- Names at least two high-priority queues with frequent exceptions (from Step 5).
- Gives at least one actionable recommendation (e.g., "prioritize clearing the backlog in [queue]" or "enforce mandatory escalation code entry").

Save the document.

## Step 7 – Validation
- Re-open `/root/Service_Queue_SLA_Audit.xlsx` and print:
  - Sheet names (must be exactly `RawData`, `Formatted Data`, `Summary`).
  - First 3 rows of each sheet.
  - Row counts for each sheet.
  - Last row of `Summary` (should be Grand Total).
- Confirm `/root/Service_Queue_SLA_Brief.docx` exists and print its text content.
- Print "DONE" when everything checks out.

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