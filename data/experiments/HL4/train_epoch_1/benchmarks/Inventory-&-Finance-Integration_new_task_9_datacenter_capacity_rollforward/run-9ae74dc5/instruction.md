# Task Instruction

Build the Excel workbook `/root/Nimbus_Capacity_Reconciliation_4-25.xlsx` following these steps:

## Step 1: Inspect all input files

Read and display the full contents of:
- `/root/compute_capacity_schedule_input.csv`
- `/root/storage_capacity_schedule_input.csv`
- `/root/capacity_ledger_balances.json`

Also read the context documents:
- `/root/Nimbus_Compute_Reservation_Register_Q1Q2_2025.txt`
- `/root/Nimbus_Storage_Commitment_Register_Q1Q2_2025.txt`
- `/root/nimbus_platform_ledger_notes_apr25.txt`

Study ALL of these carefully before writing any code. Understand what months are covered, what line items exist, what the beginning/ending balances are, and what GL balances are.

## Step 2: Understand the required structure

The workbook follows a "Harbor reconciliation" pattern:

### Detail tabs (Compute Pool #8100 and Storage Pool #8200)
- Row 1-5: Headers (pool name, column headers for months, etc.)
- Row 6+: Line items from the CSV data (these are the monthly transaction/activity rows)
- After line items, control rows in this order:
  - **Month Totals**: SUM of line item values for each month column
  - **Ending Balance**: Beginning Balance + Month Totals (rolling forward)
  - **Variance**: Ending Balance - GL Balance
  - **GL Balance**: From the JSON ledger balances
- Columns: Likely column A = row labels, columns B-M (or B-N) = individual months, column O = totals or final period summary
- The CSV files define what goes in the line item rows and month columns

### Capacity Summary tab
- B7, B8, B9 should reference column O values from one detail tab (likely Compute Pool #8100)
- B12, B13, B14 should reference column O values from the other detail tab (Storage Pool #8200)
- B16 = B9 + B14 (formula combining both pools)
- The summary likely shows: Beginning Balance, Net Activity/Month Totals, Ending Balance for each pool, then a combined total

## Step 3: Build the workbook with Python/openpyxl

Use openpyxl to create the workbook. Key rules:
- Create exactly 3 sheets in order: `Capacity Summary`, `Compute Pool #8100`, `Storage Pool #8200`
- Remove any default sheets
- Line items start at row 6 in detail tabs
- All numeric values must be stored as numbers (int or float), NOT as strings
- Control rows must use the exact names: `Month Totals`, `Ending Balance`, `Variance`, `GL Balance`
- Use Excel formulas (not hardcoded values) for Month Totals (SUM), Ending Balance, Variance, and the summary tab references
- Summary formulas in B7/B8/B9 link to column O of one detail tab; B12/B13/B14 link to column O of the other
- B16 must contain the formula `=B9+B14`
- Do NOT modify any source files

## Step 4: Validate

After creating the workbook, reopen it with openpyxl and verify:
- Sheet names and order are exactly correct
- Line items start at row 6
- Control row labels match exactly
- B16 on Capacity Summary contains formula referencing B9 and B14
- Summary cells B7/B8/B9/B12/B13/B14 contain cross-sheet references to column O
- Numeric cells are numeric type, not string
- Print the structure of each sheet for verification

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