# Task Instruction

Build an Excel workbook at `/root/Nimbus_Capacity_Reconciliation_4-25.xlsx` with exactly three sheets in this order: `Capacity Summary`, `Compute Pool #8100`, `Storage Pool #8200`.

## Step 0: Inspect all input files

1. Read and display the full contents of `/root/compute_capacity_schedule_input.csv`.
2. Read and display the full contents of `/root/storage_capacity_schedule_input.csv`.
3. Read and display the full contents of `/root/capacity_ledger_balances.json`.
4. Read and display the full contents of `/root/Nimbus_Compute_Reservation_Register_Q1Q2_2025.txt`.
5. Read and display the full contents of `/root/Nimbus_Storage_Commitment_Register_Q1Q2_2025.txt`.
6. Read and display the full contents of `/root/nimbus_platform_ledger_notes_apr25.txt`.

Do NOT proceed to building the workbook until you have read and printed every file. You need to understand the exact data, column names, row counts, and numeric values before writing any code.

## Step 1: Understand the structure

Based on the files you inspected, determine:
- What months are covered (likely Jan–Jun 2025 or similar). Each month gets a column starting from column B (or C) through the last month, with column O likely being a totals or final column.
- What line items exist in each CSV (these become the rows starting at row 6 in the detail sheets).
- What the capacity_ledger_balances.json contains (likely GL balances and/or beginning/ending balances for each pool).
- How the source .txt documents provide operational context (reservation details, commitment details, ledger notes).

The structure follows a "roll-forward" / reconciliation pattern:
- Row 1–5: Headers (sheet title, column headers for months, etc.)
- Row 6+: Line items from the CSV data
- After line items, control rows in this order: `Month Totals`, `Ending Balance`, `Variance`, `GL Balance`
- The detail sheets (Compute Pool #8100 and Storage Pool #8200) each have monthly columns and a totals column.
- Column O of each detail sheet feeds into the Capacity Summary sheet.

## Step 2: Build the detail sheets (`Compute Pool #8100` and `Storage Pool #8200`)

For each detail sheet:
1. Place appropriate headers in rows 1–5 (sheet title in row 1, month names as column headers in row 5 starting from column B, with the last data column being the rightmost month or a total column).
2. Starting at row 6, place each line item from the corresponding CSV. Each row = one line item; each column = one month's value. Keep all numeric values as Python numbers (int or float), NOT strings.
3. After all line items, add the control rows:
   - **Month Totals**: SUM formula across all line-item rows for each month column.
   - **Ending Balance**: Formula that computes the ending balance (typically Beginning Balance + Month Totals, or a cumulative roll-forward). Determine the exact formula from the data context.
   - **Variance**: Formula = Ending Balance - GL Balance.
   - **GL Balance**: Numeric value from `capacity_ledger_balances.json` for the corresponding pool.
4. Ensure column O (the 15th column, i.e., column index 15 in openpyxl) contains the summary/total values that the Capacity Summary sheet will reference.

## Step 3: Build the `Capacity Summary` sheet

This sheet summarizes both pools. The key cells are:
- **B7**: Formula referencing `='Compute Pool #8100'!O<row>` for the compute pool's relevant summary value (e.g., Month Totals or Ending Balance from column O).
- **B8**: Another compute pool reference from column O.
- **B9**: Another compute pool reference from column O.
- **B12**: Formula referencing `='Storage Pool #8200'!O<row>` for the storage pool's relevant summary value.
- **B13**: Another storage pool reference from column O.
- **B14**: Another storage pool reference from column O.
- **B16**: Formula `=B9+B14` (combines compute and storage totals).

Determine the exact row references by matching the control row positions in each detail sheet. The pattern is likely:
- B7 = Compute Ending Balance (or Month Totals) from column O
- B8 = Compute Variance from column O
- B9 = Compute GL Balance from column O
- B12–B14 = Same pattern for Storage
- B16 = B9 + B14

Adjust the exact mapping based on what you see in the data. The important contract is:
- B7, B8, B9 reference `Compute Pool #8100` column O
- B12, B13, B14 reference `Storage Pool #8200` column O
- B16 = `=B9+B14`
- All these must be Excel formulas (not hardcoded values)

## Step 4: Save and validate

1. Save the workbook to `/root/Nimbus_Capacity_Reconciliation_4-25.xlsx`.
2. Reopen it with openpyxl and verify:
   - Exactly 3 sheets in the correct order: `Capacity Summary`, `Compute Pool #8100`, `Storage Pool #8200`.
   - Line items start at row 6 in detail sheets.
   - Control rows (`Month Totals`, `Ending Balance`, `Variance`, `GL Balance`) exist with correct labels.
   - B7, B8, B9, B12, B13, B14 in Capacity Summary contain formulas referencing column O of the detail sheets.
   - B16 contains the formula `=B9+B14`.
   - All numeric cells contain numbers, not strings.
   - The source input files are unchanged.
3. Print the cell values and formulas for the key cells to confirm correctness.

## Critical constraints
- Do NOT modify any source files.
- All numeric data must be stored as numeric types in openpyxl (int/float), never as strings.
- Use Excel formulas (strings starting with `=`) for all computed cells (Month Totals sums, Ending Balance, Variance, and all Capacity Summary cross-references).
- The workbook must be saved as `.xlsx` at exactly `/root/Nimbus_Capacity_Reconciliation_4-25.xlsx`.
- Sheet names must be exactly: `Capacity Summary`, `Compute Pool #8100`, `Storage Pool #8200` — in that order.

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

Task-local resources are available under `environment/skills`: monthly-close.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=noreply@example.com, author_name=Codex Task Generator, category=cloud-finops, difficulty=medium, tags=[excel, capacity, reconciliation, rollforward, cloud-ops].
Verifier config: timeout_sec=900.0.