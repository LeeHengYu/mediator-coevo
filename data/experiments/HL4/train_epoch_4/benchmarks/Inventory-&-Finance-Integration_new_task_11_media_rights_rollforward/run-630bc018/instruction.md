# Task Instruction

## Task: Build Aurora Rights Rollforward Workbook

Create `/root/Aurora_Rights_Rollforward_4-25.xlsx` with exactly three sheets in order: `Rights Summary`, `Film Rights #2710`, `Music Rights #2720`.

### Step-by-step execution plan

#### Step 1: Inspect all input files

Read and display the contents of:
- `/root/film_rights_schedule_input.csv`
- `/root/music_rights_schedule_input.csv`
- `/root/rights_ledger_balances.json`
- `/root/Aurora_Film_Licensor_Invoices_Q1Q2_2025.txt`
- `/root/Aurora_Music_Licensor_Invoices_Q1Q2_2025.txt`
- `/root/aurora_rights_ledger_control_notes_apr25.txt`

Pay close attention to:
- Column names, number of rows, date ranges, numeric values
- The JSON structure (what keys exist, what balances are provided)
- The operational context documents for any entity names, GL account references, or balance figures
- Any month columns (likely Jan through some month) that will become columns in the detail sheets

#### Step 2: Inspect verifier

Read the verifier script (likely at `/root/verify.py` or similar path — search for `verify`, `test`, `check` files in `/root/`). Understanding the verifier is CRITICAL. Look for:
- Exact expected cell values (especially A1 of each sheet)
- Expected column headers and row labels
- Which cells are checked and what formulas/values are expected
- The control row labels: `Month Totals`, `Ending Balance`, `Variance`, `GL Balance`
- Summary sheet formula expectations at B7, B8, B9, B12, B13, B14, B16
- Any references to column O (column 15) of detail tabs

#### Step 3: Determine sheet structure from verifier + inputs

Based on the verifier and reference to the "Harbor reconciliation task" pattern:

**Detail sheets (`Film Rights #2710` and `Music Rights #2720`):**
- Cell A1: This is CRITICAL. Based on feedback from task_10, the verifier may expect A1 to be the entity/licensor name (e.g., from the input data or context docs), NOT the sheet name. Check the verifier carefully for what A1 must contain.
- Row 1: likely entity/title header
- Row 2-4: possibly header info (Beginning Balance, etc.)
- Row 5: column headers (A=description, B=some field, then month columns)
- Row 6+: line items from the CSV input
- After line items: control rows — `Month Totals` (SUM of line items per month column), `Ending Balance` (Beginning Balance + Month Totals), `Variance` (Ending Balance - GL Balance), `GL Balance` (from JSON)
- Column O (col 15): likely contains a total/annual column that the summary sheet references

**Rights Summary sheet:**
- Contains summary with references to the detail sheets
- B7 = Film detail sheet column O value (likely Ending Balance or a key total)
- B8 = another Film value from column O
- B9 = sum or derived Film value
- B12, B13, B14 = corresponding Music values
- B16 = B9 + B14 (combined total)

#### Step 4: Build the workbook with Python + openpyxl

Write a Python script that:
1. Reads both CSVs with pandas
2. Reads the JSON ledger balances
3. Creates the workbook with exactly 3 sheets in the required order
4. Populates detail sheets following the discovered structure:
   - Line items start at row 6
   - All numeric values stored as numbers (int/float), NOT strings
   - Control rows after line items with proper formulas or computed values
   - Month Totals = SUM of line item values per column
   - Ending Balance = Beginning Balance + Month Totals  
   - GL Balance from JSON
   - Variance = Ending Balance - GL Balance
5. Populates Rights Summary with cell references/formulas pointing to column O of detail tabs
6. B16 formula must be `=B9+B14`
7. Saves to `/root/Aurora_Rights_Rollforward_4-25.xlsx`

#### Step 5: Validate

- Re-open the workbook and verify sheet names, order, key cell values
- Run the verifier script if found
- If verifier fails, read the error output carefully, fix the specific failing assertions, and re-run

### Critical reminders
- A1 of detail sheets: CHECK THE VERIFIER for exact expected value. Do NOT assume it equals the sheet name.
- All numeric cells must contain numeric types, not strings
- Do not modify any source files
- Column O = column index 15 (1-indexed) in openpyxl
- Summary formulas in B7/B8/B9/B12/B13/B14 should reference column O of detail tabs using cross-sheet references like `='Film Rights #2710'!O<row>`
- B16 must be `=B9+B14`

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