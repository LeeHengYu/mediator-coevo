# Task Instruction

## Task: Build Aurora Rights Rollforward Workbook

Create an Excel workbook at `/root/Aurora_Rights_Rollforward_4-25.xlsx` with exactly three sheets in order: `Rights Summary`, `Film Rights #2710`, `Music Rights #2720`.

### Step 1: Inspect all input files thoroughly

1. `cat /root/film_rights_schedule_input.csv` — note every column, every row, all amounts, dates, vendor names, descriptions.
2. `cat /root/music_rights_schedule_input.csv` — same.
3. `cat /root/rights_ledger_balances.json` — note all balances (beginning, ending, GL balances, etc.).
4. `cat /root/Aurora_Film_Licensor_Invoices_Q1Q2_2025.txt` — read for operational context: vendor/licensor names, entity name (e.g., "Aurora Stream" or similar), any titles or headers that should appear in the workbook.
5. `cat /root/Aurora_Music_Licensor_Invoices_Q1Q2_2025.txt` — same.
6. `cat /root/aurora_rights_ledger_control_notes_apr25.txt` — read for any control totals, GL account references, entity names.

### Step 2: Inspect the verifier

Look for any test or verifier script in the task directory. Run:
```
find / -path '*/new_task_11_media_rights_rollforward*' -type f 2>/dev/null
```
Also check:
```
find / -name 'test_outputs*' -type f 2>/dev/null
find / -name '*.test.*' -type f 2>/dev/null
find / -name 'verify*' -type f 2>/dev/null
```
Read any verifier/test file completely. This is critical — the verifier defines the exact contract (cell values, formulas, sheet names, header strings). Extract every assertion and use them as hard requirements.

### Step 3: Understand the reference structure

The workbook follows a "Harbor reconciliation" pattern:
- **Rows 1-5**: Header area (Row 1 = title/entity name, Row 3-5 = column headers)
- **Row 6 onward**: Line items (one per transaction/invoice)
- **Control rows** after line items: `Month Totals`, `Ending Balance`, `Variance`, `GL Balance`
- **Columns**: Likely A = description/vendor, B-N = monthly columns (or date/amount columns), O = total/summary column
- The `Rights Summary` sheet has formulas in specific cells:
  - B7, B8, B9 link to column O of `Film Rights #2710`
  - B12, B13, B14 link to column O of `Music Rights #2720`
  - B16 = B9 + B14

### Step 4: Build the workbook with Python + openpyxl

Write a Python script that:

1. Reads the CSV files with the `csv` module (preserve numeric types — convert numeric strings to float/int).
2. Reads the JSON file.
3. Creates the workbook with `openpyxl`.

#### Detail sheets (`Film Rights #2710` and `Music Rights #2720`):
- **A1**: Set to the entity/company name found in the source documents (likely "Aurora Stream" or whatever the invoices/notes specify — check verifier assertions for exact string). This is CRITICAL — a wrong A1 value will fail the verifier.
- **Row 1-5**: Title, subtitle, column headers as appropriate. Check verifier for exact expected values.
- **Row 6+**: Line items from the corresponding CSV. Each numeric value must be stored as a number, not a string.
- **After line items**, add control rows in column A:
  - `Month Totals` — with SUM formulas across line item rows for each column
  - `Ending Balance` — formula or value as appropriate
  - `Variance` — formula (Ending Balance minus GL Balance, or as defined)
  - `GL Balance` — from the JSON ledger balances
- **Column O**: Should contain row totals or key summary values that the Rights Summary sheet references.

#### `Rights Summary` sheet:
- Structure with labels in column A and values/formulas in column B.
- B7 = reference to Film Rights detail sheet column O (e.g., `='Film Rights #2710'!O<row>` for the relevant control row)
- B8 = similar reference
- B9 = similar reference  
- B12 = reference to Music Rights detail sheet column O
- B13 = similar reference
- B14 = similar reference
- B16 = `=B9+B14`
- Use `openpyxl` formula strings (e.g., `ws['B7'] = "='Film Rights #2710'!O15"`).

### Step 5: Validate

1. After creating the workbook, re-open it with openpyxl and verify:
   - Exactly 3 sheets in correct order
   - Sheet names match exactly
   - A1 of detail sheets has the correct entity name
   - Line items start at row 6
   - Control rows exist with correct labels
   - Numeric cells are numeric (not strings)
   - Summary formulas are present in the Rights Summary sheet
2. Run the verifier/test if found.
3. If the verifier fails, read the exact error, fix, and re-run.

### Key Warnings (from cross-task feedback):
- **A1 of detail sheets must contain the exact entity/company name the verifier expects.** Read the verifier assertions to find this string. Do NOT use a generic placeholder.
- **All numeric values must be stored as numbers**, not text strings.
- **Do not modify source files.**
- **Sheet names must be exact** (including spaces, capitalization, and `#` symbols).
- **Formula cells must contain formula strings**, not computed values, especially B16 = `=B9+B14`.

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

Task-local resources are available under `environment/skills`: invoice-organizer, monthly-close.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=noreply@example.com, author_name=Codex Task Generator, category=media-operations, difficulty=medium, tags=[excel, media-rights, invoice-normalization, reconciliation, rollforward].
Verifier config: timeout_sec=900.0.