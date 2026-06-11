# Task Instruction

Write and execute a single Python script that:

1. **Install dependencies** (if needed): `pip install openpyxl python-docx`

2. **Read the source workbook** `/root/Ticket_Queue.xlsx`:
   - Read the `Tickets` sheet into a list of rows (preserve headers and all data).
   - Read the `SLA_Rules` sheet into a lookup dictionary keyed by `Priority Tier` (or whatever the priority column is named), storing `Max Open Hours` and `Escalation Required` for each tier.
   - Print the SLA_Rules data and the first few Tickets rows for verification.

3. **Create `/root/Service_Queue_SLA_Audit.xlsx`** with exactly three sheets:

   **Sheet `RawData`:**
   - Copy the `Tickets` table exactly (all headers and rows, same order).

   **Sheet `Formatted Data`:**
   - Same row order as RawData.
   - First 8 columns exactly: `Ticket ID`, `Queue`, `Priority Tier`, `Open Age Hours`, `Owner`, `Escalation Code`, `Region`, `Analyst`.
   - Add 4 new columns (9–12) with headers: `SLA Breach`, `Missing Escalation`, `Total Errors`, `Error Summary`.
   - Compute concrete values (not formulas):
     - `SLA Breach` = 1 if the row's `Open Age Hours` > `Max Open Hours` for that row's `Priority Tier` (from SLA_Rules), else 0.
     - `Missing Escalation` = 1 if `Escalation Required` is `Y` for that tier AND the row's `Escalation Code` is blank/None/empty, else 0.
     - `Total Errors` = SLA Breach + Missing Escalation.
     - `Error Summary` = one of exactly: `None`, `SLA Breach`, `Missing Escalation`, `SLA Breach, Missing Escalation` (based on which flags are 1).
   - Write all values as concrete numbers/strings.

   **Sheet `Summary`:**
   - Headers exactly: `Queue`, `Region`, `SLA Breaches`, `Missing Escalations`, `Total Errors`.
   - Aggregate from `Formatted Data` by (Queue, Region).
   - Include only groups where Total Errors > 0.
   - Sort by Queue ascending, then Region ascending.
   - Append a final row: Queue=`Grand Total`, Region=`-`, and the dataset-wide totals for the three numeric columns.

4. **Create `/root/Service_Queue_SLA_Brief.docx`:**
   - Write a 3–6 sentence executive summary paragraph that:
     - Defines both checks in plain language (SLA Breach = ticket open longer than the allowed max hours for its priority tier; Missing Escalation = escalation was required by SLA rules but no escalation code was recorded).
     - States the computed totals for SLA Breaches, Missing Escalations, and Total Errors.
     - Gives at least one actionable recommendation.
     - Mentions at least two specific queues that have the highest error counts.

5. **Validate outputs:**
   - Reopen `/root/Service_Queue_SLA_Audit.xlsx` and print sheet names, row counts per sheet, and the first 3 data rows of `Formatted Data` and `Summary`.
   - Confirm `/root/Service_Queue_SLA_Brief.docx` exists and print its text content.

**Important details:**
- Column name matching: inspect the actual headers in `Tickets` and `SLA_Rules` sheets before processing. Map them to the required output column names. Handle case and whitespace variations.
- For `Escalation Code` blank check: treat None, empty string, and whitespace-only as blank.
- Use openpyxl for Excel I/O and python-docx for Word.
- File and sheet names must be exactly as specified.
- Do not use pandas (to avoid dtype issues); use openpyxl directly for reading and writing.

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