# Task Instruction

Build an Excel workbook at `/root/Nimbus_Capacity_Reconciliation_4-25.xlsx` following these steps:

## Step 1: Inspect all input files

Read and print the contents of:
- `/root/compute_capacity_schedule_input.csv`
- `/root/storage_capacity_schedule_input.csv`
- `/root/capacity_ledger_balances.json`
- `/root/Nimbus_Compute_Reservation_Register_Q1Q2_2025.txt`
- `/root/Nimbus_Storage_Commitment_Register_Q1Q2_2025.txt`
- `/root/nimbus_platform_ledger_notes_apr25.txt`

Understand the data structure, column headers, date ranges (likely months Jan-Jun or similar), line items, and what the beginning/ending balances and GL balances are.

## Step 2: Build the workbook with openpyxl

Create exactly three sheets in this order:
1. `Capacity Summary`
2. `Compute Pool #8100`
3. `Storage Pool #8200`

### Detail Sheets (`Compute Pool #8100` and `Storage Pool #8200`):

These follow a rollforward schedule pattern:
- Row 1-5: Headers (Row 1 = sheet title, Row 3 or 4 = month column headers, Row 5 = column labels like 'Description' in A, months in B through N or O, 'Total' in the last column)
- Row 6+: Line items from the CSV data. Each row is a capacity line item; columns are months with numeric values.
- After line items, insert control rows in this exact order:
  - `Month Totals` — SUM of the line-item rows above for each month column
  - `Ending Balance` — derived from Beginning Balance + Month Totals (or as the data dictates)
  - `Variance` — Ending Balance minus GL Balance
  - `GL Balance` — from the JSON ledger balances
- The structure must have a `Beginning Balance` row (likely at row 5 or as first line item) and the control rows after the line items.
- Column O (or the last data column) should contain totals or the final month's values — these are what the Summary sheet references.

### `Capacity Summary` Sheet:

This sheet summarizes both pools. The structure should be:
- Row 6 area: Compute pool section header
- B7 = reference to `Compute Pool #8100` column O `Ending Balance` cell
- B8 = reference to `Compute Pool #8100` column O `GL Balance` cell  
- B9 = reference to `Compute Pool #8100` column O `Variance` cell
- Row 11 area: Storage pool section header
- B12 = reference to `Storage Pool #8200` column O `Ending Balance` cell
- B13 = reference to `Storage Pool #8200` column O `GL Balance` cell
- B14 = reference to `Storage Pool #8200` column O `Variance` cell
- B16 = B9 + B14 (combined variance, must be an Excel formula: `=B9+B14`)

**IMPORTANT**: Examine the CSV files carefully to determine the exact row positions of each control row. The formulas in B7/B8/B9/B12/B13/B14 must reference the correct rows in column O of the detail tabs. Adapt the mapping based on actual data.

## Step 3: Ensure correctness

- All numeric values must be stored as numbers (int or float), NOT strings.
- Use Excel formulas (not hardcoded values) for: Month Totals (SUM), Ending Balance, Variance, and all Summary sheet references.
- B16 must be the formula `=B9+B14`.
- Do NOT modify any source files.
- Remove the default 'Sheet' if openpyxl creates one.
- Save to exactly `/root/Nimbus_Capacity_Reconciliation_4-25.xlsx`.

## Step 4: Validate

After saving, reopen the workbook and verify:
- Exactly 3 sheets with exact names in order
- Line items start at row 6 in detail sheets
- Control rows exist with correct labels
- Summary cell B16 contains formula `=B9+B14`
- Summary cells B7-B14 contain cross-sheet references to column O
- All numeric cells contain numbers, not strings
- Print key cell values and formulas for confirmation

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