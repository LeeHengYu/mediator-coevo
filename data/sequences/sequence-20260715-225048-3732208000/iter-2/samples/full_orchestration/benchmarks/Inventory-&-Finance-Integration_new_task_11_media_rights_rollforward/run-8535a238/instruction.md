# Task Instruction

## Task: Build Aurora Rights Rollforward Excel Workbook

Create `/root/Aurora_Rights_Rollforward_4-25.xlsx` with exactly 3 sheets in order: `Rights Summary`, `Film Rights #2710`, `Music Rights #2720`.

### Step 0: Inspect All Input Files

1. Read and display `/root/film_rights_schedule_input.csv` completely.
2. Read and display `/root/music_rights_schedule_input.csv` completely.
3. Read and display `/root/rights_ledger_balances.json` completely.
4. Read `/root/Aurora_Film_Licensor_Invoices_Q1Q2_2025.txt` for operational context.
5. Read `/root/Aurora_Music_Licensor_Invoices_Q1Q2_2025.txt` for operational context.
6. Read `/root/aurora_rights_ledger_control_notes_apr25.txt` for operational context.
7. List all files in `/root/` to check if there is any reference/template workbook.

Do NOT proceed to building the workbook until you have fully read and understood every input file.

### Step 1: Understand the Structure (Harbor-style Reconciliation)

The workbook follows a "Harbor reconciliation" pattern:
- **Rows 1-5**: Header area (title, column headers, etc.)
- **Row 1**: Sheet title (e.g., cell A1 = sheet name or descriptive title)
- **Row 5**: Column headers for the data table
- **Row 6 onward**: Line item data rows (one per rights deal/license)
- After all line items, 4 control rows in order:
  1. `Month Totals` — sums of each monthly column for the line items above
  2. `Ending Balance` — derived from Beginning Balance + Month Totals (or as appropriate)
  3. `Variance` — Ending Balance minus GL Balance
  4. `GL Balance` — from the ledger balances JSON

Columns should include:
- Column A: Deal/license identifier or description
- Column B: Beginning Balance
- Columns C through N: Monthly amounts (Jan through Dec, or the relevant months)
- Column O: Total / Ending amount (sum across months + beginning balance, or as the data dictates)

Examine the CSV files carefully to determine the exact column structure. The CSVs are "normalized input" so their columns define the detail sheet structure.

### Step 2: Build Detail Sheet — `Film Rights #2710`

1. Set A1 = `Film Rights #2710` (the sheet tab name must also be exactly this).
2. Populate header rows (rows 1-5) with appropriate titles and column headers derived from the CSV columns.
3. Starting at row 6, populate one row per record from `film_rights_schedule_input.csv`.
4. All numeric values must be stored as numbers (int or float), NOT as strings.
5. After the last line item row, add the 4 control rows:
   - `Month Totals`: Column A = "Month Totals", each numeric column = SUM of that column's line item cells (use Excel SUM formulas).
   - `Ending Balance`: Column A = "Ending Balance". This should be a formula: Beginning Balance (from ledger JSON for account 2710) + the Month Totals row's total, OR as dictated by the data structure. Column O should contain the ending balance value.
   - `Variance`: Column A = "Variance". Column O (and other relevant columns) = Ending Balance - GL Balance (formula).
   - `GL Balance`: Column A = "GL Balance". Populate from `rights_ledger_balances.json` for account 2710.

### Step 3: Build Detail Sheet — `Music Rights #2720`

Same structure as Film Rights but using `music_rights_schedule_input.csv` and account 2720 from the JSON.

### Step 4: Build `Rights Summary` Sheet

This is the FIRST sheet (leftmost tab). Structure:
- Row 1: Title
- Rows 2-5: Headers/labels
- The summary uses specific cells with formulas:
  - **B7** = link to `Film Rights #2710` column O Ending Balance (or Month Totals — determine from context)
  - **B8** = link to `Film Rights #2710` column O GL Balance (or relevant control row)
  - **B9** = link to `Film Rights #2710` column O Variance (or computed from B7-B8)
  - **B12** = link to `Music Rights #2720` column O Ending Balance
  - **B13** = link to `Music Rights #2720` column O GL Balance
  - **B14** = link to `Music Rights #2720` column O Variance
  - **B16** = `=B9+B14` (combined variance, must be this exact formula)

Use cross-sheet references like `='Film Rights #2710'!O<row>` for the links.

Label column A appropriately:
- A7: Film Rights Ending Balance (or similar)
- A8: Film Rights GL Balance
- A9: Film Rights Variance
- A12: Music Rights Ending Balance
- A13: Music Rights GL Balance
- A14: Music Rights Variance
- A16: Total Variance (or Combined Variance)

### Step 5: Validation

After creating the workbook:
1. Re-open it with openpyxl and verify:
   - Exactly 3 sheets in order: `Rights Summary`, `Film Rights #2710`, `Music Rights #2720`
   - Sheet names are EXACTLY as specified (case-sensitive, including spaces and #)
   - A1 of each detail sheet contains the sheet's title
   - Line items start at row 6 in detail sheets
   - Control rows exist with correct labels in column A
   - B16 in Rights Summary contains a formula `=B9+B14` (not a hardcoded value)
   - B7, B8, B9, B12, B13, B14 contain cross-sheet formula references
   - All numeric data cells contain numbers, not strings
   - Column O of detail sheets has the values that the summary sheet references
2. Print the sheet names, cell A1 of each sheet, the control row labels, and the formula cells from the summary sheet.
3. Fix any issues found.

### Critical Rules
- Do NOT modify any source files.
- All numeric values must be numeric type in Excel (use int/float, not str).
- Use `openpyxl` for creating the workbook.
- The final file must be at exactly `/root/Aurora_Rights_Rollforward_4-25.xlsx`.
- Pay very careful attention to exact sheet names — the verifier will check them character by character.
- Pay attention to the A1 cell content of each sheet — it was a failure point in similar tasks.
- Formulas should use Excel formula syntax (strings starting with `=`), not computed values, for the summary links and control row calculations.

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