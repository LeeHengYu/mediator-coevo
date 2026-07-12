# Task Instruction

Build an Excel workbook at `/root/MetroLink_Pass_Liability_4-25.xlsx` with exactly three sheets in this order: `Transit Summary`, `Bus Program #4310`, `Rail Program #4320`.

## Step 0: Inspect all input files

1. Read and display `/root/bus_pass_schedule_input.csv` completely.
2. Read and display `/root/rail_pass_schedule_input.csv` completely.
3. Read and display `/root/fare_liability_balances.json` completely.
4. Read `/root/MetroLink_Bus_Pass_Issuance_Notes_Q1Q2_2025.txt` for operational context.
5. Read `/root/MetroLink_Rail_Pass_Issuance_Notes_Q1Q2_2025.txt` for operational context.
6. Read `/root/metrolink_fare_ledger_control_notes_apr25.txt` for operational context.
7. Check if there are any other files in `/root/` that might be reference examples (e.g., any existing `.xlsx` files). List all files in `/root/`.

## Step 1: Understand the data structure

After reading the files, identify:
- The column structure of each CSV (likely months as columns, with line items as rows)
- The opening/ending balances and GL balances from the JSON file
- Any variance or control values mentioned in the context notes

## Step 2: Build the two detail sheets (`Bus Program #4310` and `Rail Program #4320`)

For each detail sheet, follow this structure:
- Row 1: Sheet title (e.g., `Bus Program #4310`)
- Row 2-4: Header area (program info, column headers, etc.)
- Row 5: Column headers — Column A = line item description, Columns B through N (or similar) = individual months, Column O = Total/Summary column
- Row 6 onward: Line items from the corresponding CSV, starting at row 6
- After line items, include these control rows in order:
  - `Month Totals` — sum of all line item values for each month column
  - `Ending Balance` — computed as Opening Balance + Month Totals (or as specified by the data)
  - `Variance` — difference between Ending Balance and GL Balance
  - `GL Balance` — from the JSON balances file

Column O should contain totals/summaries across all months for each row. Use SUM formulas across the month columns for each line item row in column O. The control rows in column O should also use appropriate formulas.

All numeric values must be stored as numbers, not text strings.

## Step 3: Build the `Transit Summary` sheet

This sheet summarizes both programs. The structure must have:
- A header area in rows 1-5
- Row 6 area: labels for Bus Program and Rail Program sections
- Key cells with formulas linking to column O of the detail tabs:
  - B7 = `='Bus Program #4310'!O{Month Totals row}` (Bus Month Totals from col O)
  - B8 = `='Bus Program #4310'!O{Ending Balance row}` (Bus Ending Balance from col O)
  - B9 = `='Bus Program #4310'!O{Variance row}` (Bus Variance from col O)
  - B12 = `='Rail Program #4320'!O{Month Totals row}` (Rail Month Totals from col O)
  - B13 = `='Rail Program #4320'!O{Ending Balance row}` (Rail Ending Balance from col O)
  - B14 = `='Rail Program #4320'!O{Variance row}` (Rail Variance from col O)
  - B16 = `=B9+B14` (Combined Variance)

The exact row references for Month Totals, Ending Balance, and Variance on the detail sheets depend on how many line items exist. Determine these row numbers after loading the CSV data.

Label column A appropriately:
- A7: Bus Month Totals (or similar)
- A8: Bus Ending Balance
- A9: Bus Variance
- A10-A11: spacer/blank
- A12: Rail Month Totals
- A13: Rail Ending Balance
- A14: Rail Variance
- A15: spacer/blank
- A16: Combined Variance

## Step 4: Populate data

Use openpyxl (install if needed via `pip install openpyxl`) to create the workbook programmatically in Python.

- Parse both CSVs with the csv module or pandas.
- Parse the JSON file for balance information.
- Map the CSV data into the detail sheets starting at row 6.
- Compute or place formulas for Month Totals (SUM of line items), Ending Balance (Opening Balance + Month Totals or as data dictates), GL Balance (from JSON), and Variance (Ending Balance - GL Balance).
- Use Excel formulas (strings starting with `=`) for all computed cells, not hardcoded values, especially for Month Totals, Ending Balance, Variance, and all Transit Summary references.
- Ensure column O on detail sheets contains SUM formulas across the month columns.

## Step 5: Validate

1. Re-open the saved workbook with openpyxl and verify:
   - Exactly 3 sheets exist with exact names: `Transit Summary`, `Bus Program #4310`, `Rail Program #4320` in that order.
   - Line items start at row 6 on detail sheets.
   - Control rows (`Month Totals`, `Ending Balance`, `Variance`, `GL Balance`) exist with those exact labels in column A.
   - B7, B8, B9, B12, B13, B14 on Transit Summary contain formulas referencing column O of the detail tabs.
   - B16 on Transit Summary contains `=B9+B14`.
   - All numeric cells are numeric type, not strings.
   - The file is saved at `/root/MetroLink_Pass_Liability_4-25.xlsx`.
2. Print the cell values and types for the key cells to confirm.
3. Do NOT modify any source input files.

## Important Notes
- Use `openpyxl` to write `.xlsx` format.
- Store all monetary/numeric values as Python floats or ints, never as strings.
- Use Excel formula strings (e.g., `=SUM(B6:B15)`) for computed cells.
- The sheet order matters: Transit Summary must be first (index 0), Bus second (index 1), Rail third (index 2).
- If any input file structure is unexpected, adapt the mapping but preserve the required output structure (line items at row 6, control rows after, summary formulas in specified cells).

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

Task-local resources are available under `environment/skills`: expense-tracker, monthly-close.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=noreply@example.com, author_name=Codex Task Generator, category=transit-operations, difficulty=medium, tags=[excel, public-transit, subsidy, reconciliation, program-tracking].
Verifier config: timeout_sec=900.0.