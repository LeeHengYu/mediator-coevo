# Task Instruction

## Task: Service Queue SLA Audit

You must read `/root/Ticket_Queue.xlsx` and produce two deliverables:
1. `/root/Service_Queue_SLA_Audit.xlsx`
2. `/root/Service_Queue_SLA_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect the source data
- Open `/root/Ticket_Queue.xlsx` using openpyxl (or pandas).
- Read the `Tickets` sheet. Print the column headers and first 5 rows to understand the schema.
- Read the `SLA_Rules` sheet. Print all rows. Identify the columns for `Priority Tier`, `Max Open Hours`, and `Escalation Required`.
- Note: The column names in the source may differ slightly (e.g., spaces, casing). Use the actual column names from the file.

#### Step 1: Build a Python script that does everything below

Use `openpyxl` for Excel writing and `python-docx` for Word. Install them if needed (`pip install openpyxl python-docx`).

#### Step 2: Create `RawData` sheet
- Copy the entire `Tickets` sheet exactly (headers + all data rows) into a worksheet named `RawData` in the output workbook.

#### Step 3: Create `Formatted Data` sheet
- Copy the same rows from `Tickets` but keep only the first 8 columns in this exact order and with these exact header names:
  1. `Ticket ID`
  2. `Queue`
  3. `Priority Tier`
  4. `Open Age Hours`
  5. `Owner`
  6. `Escalation Code`
  7. `Region`
  8. `Analyst`
- Map source columns to these names. If the source uses different names, map them correctly based on content inspection.
- Build a lookup dictionary from `SLA_Rules`: for each `Priority Tier`, store `Max Open Hours` (numeric) and `Escalation Required` (string, 'Y' or 'N').
- For each row, compute:
  - `SLA Breach`: 1 if `Open Age Hours` > `Max Open Hours` for that row's `Priority Tier`, else 0. (Use strict greater-than.)
  - `Missing Escalation`: 1 if the SLA rule says `Escalation Required` == 'Y' for that tier AND the row's `Escalation Code` is blank/None/empty string, else 0.
  - `Total Errors`: `SLA Breach + Missing Escalation`
  - `Error Summary`: Exactly one of these strings:
    - `"None"` if Total Errors == 0
    - `"SLA Breach"` if only SLA Breach == 1
    - `"Missing Escalation"` if only Missing Escalation == 1
    - `"SLA Breach, Missing Escalation"` if both == 1
- Write these as concrete values (integers and strings), NOT formulas.
- The column headers for columns 9-12 must be exactly: `SLA Breach`, `Missing Escalation`, `Total Errors`, `Error Summary`.

#### Step 4: Create `Summary` sheet
- From the `Formatted Data` rows, group by `(Queue, Region)`.
- For each group, sum `SLA Breach` (as `SLA Breaches`), `Missing Escalation` (as `Missing Escalations`), and `Total Errors`.
- Keep only groups where `Total Errors > 0`.
- Sort by `Queue` ascending then `Region` ascending (standard alphabetical).
- Headers must be exactly: `Queue`, `Region`, `SLA Breaches`, `Missing Escalations`, `Total Errors`.
- Append a final row: `Queue` = `Grand Total`, `Region` = `-`, and the remaining columns are the dataset-wide totals (sum across ALL rows in Formatted Data, not just the filtered groups — but since groups with 0 errors contribute 0, summing the filtered groups gives the same result).

#### Step 5: Save the Excel file
- Save as `/root/Service_Queue_SLA_Audit.xlsx`.
- Ensure the sheet order is: `RawData`, `Formatted Data`, `Summary`.

#### Step 6: Create Word document `/root/Service_Queue_SLA_Brief.docx`
- Write an executive summary paragraph (3-6 sentences) that:
  - Defines both checks in plain language: "An SLA Breach occurs when a ticket's open age exceeds the maximum allowed hours for its priority tier" and "A Missing Escalation occurs when a ticket's priority tier requires escalation but no escalation code has been assigned."
  - States the computed totals: total SLA Breaches, total Missing Escalations, total Total Errors (use the actual numbers from the Grand Total row).
  - Mentions at least two specific queues that have the highest error counts (look at the Summary data to identify them).
  - Includes at least one actionable recommendation (e.g., "We recommend implementing automated escalation routing for high-priority tiers and conducting weekly SLA compliance reviews for the identified queues.").
- Save the file.

#### Step 7: Validate
- Re-open `/root/Service_Queue_SLA_Audit.xlsx` and verify:
  - Sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
  - `RawData` row count matches source `Tickets` row count.
  - `Formatted Data` has 12 columns with correct headers.
  - `Summary` last row has `Queue` == `Grand Total`.
  - Print the Summary sheet contents for verification.
- Confirm `/root/Service_Queue_SLA_Brief.docx` exists and print its text content.

### Important Notes
- Do NOT hardcode SLA thresholds. Read them from the `SLA_Rules` sheet.
- Use exact filenames and sheet names as specified.
- When checking if `Escalation Code` is blank, handle None, empty string, and whitespace-only values.
- `Open Age Hours` comparison must be strictly greater than (not >=) `Max Open Hours`.
- Write all computed columns as static values, not Excel formulas.

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