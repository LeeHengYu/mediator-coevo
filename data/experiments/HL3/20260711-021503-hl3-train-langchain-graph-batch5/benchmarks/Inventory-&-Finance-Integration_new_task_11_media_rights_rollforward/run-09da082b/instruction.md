# Task Instruction

Build an Excel workbook at `/root/Aurora_Rights_Rollforward_4-25.xlsx` with exactly three sheets. Follow these steps carefully:

## Step 1: Inspect All Input Files

Read and understand the contents of:
- `/root/film_rights_schedule_input.csv`
- `/root/music_rights_schedule_input.csv`
- `/root/rights_ledger_balances.json`

Also read the operational context files for any additional details:
- `/root/Aurora_Film_Licensor_Invoices_Q1Q2_2025.txt`
- `/root/Aurora_Music_Licensor_Invoices_Q1Q2_2025.txt`
- `/root/aurora_rights_ledger_control_notes_apr25.txt`

Pay close attention to column names, date formats, numeric values, and any account numbers or categories. Understand what data maps to which detail sheet.

## Step 2: Understand the Reconciliation Structure

Each detail sheet (`Film Rights #2710` and `Music Rights #2720`) follows this layout:
- **Row 1-5**: Headers / column labels. The columns likely span A through O (or more), where column O contains monthly or period totals that the summary references.
- **Row 6 onward**: Line item data rows (one per rights item/transaction).
- After all line items, there are exactly four control rows in this order:
  1. `Month Totals` — sums of the line item amounts per column
  2. `Ending Balance` — computed ending balance (likely opening balance + month totals)
  3. `Variance` — difference between Ending Balance and GL Balance
  4. `GL Balance` — the general ledger balance from `rights_ledger_balances.json`

Column structure likely includes months as columns (e.g., columns for Jan through some end month), with column O being a total or final-period column.

## Step 3: Build the Detail Sheets

For each detail sheet:
1. **Headers (rows 1-5)**: Create appropriate headers. Row 1 could be the sheet title. Include column headers for the line items — likely: a description/name column (A), an opening balance column (B), monthly columns, and a total column ending at or around column O.
2. **Line items (starting row 6)**: Populate from the corresponding CSV. Each row is a rights item. Ensure all monetary values are stored as **numbers, not text**. Parse any currency strings to floats.
3. **Month Totals row**: After the last line item, create a row labeled `Month Totals` in column A. Use SUM formulas to total each numeric column across the line item rows.
4. **Ending Balance row**: Label `Ending Balance` in column A. This should be the opening/beginning balance + Month Totals (use a formula).
5. **GL Balance row**: Label `GL Balance` in column A. Pull the appropriate value from `rights_ledger_balances.json` (account #2710 for Film, #2720 for Music). Store as a number.
6. **Variance row**: Label `Variance` in column A. Formula = Ending Balance - GL Balance (for each column, or at minimum column O).

**Important**: The exact row numbers of control rows depend on how many line items exist. The control rows must appear in the order: Month Totals, Ending Balance, Variance, GL Balance — immediately after the last line item.

## Step 4: Build the Rights Summary Sheet

The `Rights Summary` sheet must be the **first** sheet in the workbook. Structure:
- It summarizes both detail tabs.
- Key cells with formulas:
  - **B7**: Links to `'Film Rights #2710'!O__` (the Ending Balance cell in column O of the Film tab) — or another relevant Film summary value. Inspect the data to determine whether B7/B8/B9 map to Month Totals/Ending Balance/Variance or GL Balance/Ending Balance/Variance for Film.
  - **B8**: Another Film summary link from column O
  - **B9**: Another Film summary link from column O
  - **B12**: Links to `'Music Rights #2720'!O__` (corresponding Music summary value)
  - **B13**: Another Music summary link from column O
  - **B14**: Another Music summary link from column O
  - **B16**: Formula `=B9+B14` (combined total from both tabs)

The most likely mapping (based on standard rollforward reconciliation):
- B7 = Film Ending Balance (from column O of Film detail tab's Ending Balance row)
- B8 = Film GL Balance (from column O of Film detail tab's GL Balance row)
- B9 = Film Variance (from column O of Film detail tab's Variance row)
- B12 = Music Ending Balance (from column O of Music detail tab's Ending Balance row)
- B13 = Music GL Balance (from column O of Music detail tab's GL Balance row)
- B14 = Music Variance (from column O of Music detail tab's Variance row)
- B16 = B9 + B14 (Total Variance)

However, **inspect the input data and context files first** to confirm the correct mapping. Add appropriate labels in column A for these rows.

## Step 5: Sheet Order

The workbook must have sheets in exactly this order:
1. `Rights Summary`
2. `Film Rights #2710`
3. `Music Rights #2720`

## Step 6: Validation

After creating the workbook:
1. Re-open it with openpyxl and verify:
   - Exactly 3 sheets in the correct order with exact names
   - Line items start at row 6 in detail sheets
   - Control rows exist with exact labels: `Month Totals`, `Ending Balance`, `Variance`, `GL Balance`
   - B7, B8, B9, B12, B13, B14 in Rights Summary contain cross-sheet formulas referencing column O
   - B16 contains formula `=B9+B14`
   - All monetary values are numeric (int or float), not strings
2. Print the sheet names, a sample of cell values from each sheet, and the formulas in the summary cells.

## Constraints
- Do NOT modify any source input files.
- Use `openpyxl` to create the workbook (install if needed).
- All numeric values must be stored as Python numbers (int/float), never as strings.
- The final file must be at exactly `/root/Aurora_Rights_Rollforward_4-25.xlsx`.

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