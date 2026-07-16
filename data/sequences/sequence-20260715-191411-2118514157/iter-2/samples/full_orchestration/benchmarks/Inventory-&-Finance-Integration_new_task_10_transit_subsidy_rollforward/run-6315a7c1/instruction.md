# Task Instruction

Build the Excel workbook `/root/MetroLink_Pass_Liability_4-25.xlsx` by following these steps precisely:

## Step 1: Inspect all input files

```bash
cat /root/bus_pass_schedule_input.csv
cat /root/rail_pass_schedule_input.csv
cat /root/fare_liability_balances.json
cat /root/MetroLink_Bus_Pass_Issuance_Notes_Q1Q2_2025.txt
cat /root/MetroLink_Rail_Pass_Issuance_Notes_Q1Q2_2025.txt
cat /root/metrolink_fare_ledger_control_notes_apr25.txt
```

Read every file carefully before writing any code. Understand the data structure, column names, date ranges, and balances.

## Step 2: Understand the required workbook structure

The workbook must have exactly 3 sheets in this order:
1. `Transit Summary`
2. `Bus Program #4310`
3. `Rail Program #4320`

### Detail sheets (`Bus Program #4310` and `Rail Program #4320`)
- These are rollforward/reconciliation schedules.
- Line items (monthly data rows) start at **row 6**.
- Columns likely represent months (with column O being a total or final month column — inspect the CSV to determine the exact column layout).
- After the monthly line-item rows, there must be these control rows in order:
  - `Month Totals` — sums of the line-item rows above for each column
  - `Ending Balance` — computed from beginning balance + totals (or as the rollforward formula dictates)
  - `Variance` — difference between ending balance and GL balance
  - `GL Balance` — from the fare_liability_balances.json or ledger control notes
- All numeric values must be stored as numbers, not text strings.

### Summary sheet (`Transit Summary`)
- This sheet summarizes data from both detail tabs.
- The following cells must contain formulas (not hardcoded values):
  - **B7** = links to column O of `Bus Program #4310` (likely the Ending Balance or a key total)
  - **B8** = links to column O of `Bus Program #4310` (another key row)
  - **B9** = links to column O of `Bus Program #4310` (another key row)
  - **B12** = links to column O of `Rail Program #4320` (corresponding row)
  - **B13** = links to column O of `Rail Program #4320` (corresponding row)
  - **B14** = links to column O of `Rail Program #4320` (corresponding row)
  - **B16** = `B9 + B14` (combined total from both programs)
- The exact mapping of B7/B8/B9 to which control rows in the Bus tab, and B12/B13/B14 to which control rows in the Rail tab, should be inferred from the data context. A reasonable mapping is:
  - B7 → Bus Month Totals (col O), B8 → Bus Ending Balance (col O), B9 → Bus Variance or GL Balance (col O)
  - B12 → Rail Month Totals (col O), B13 → Rail Ending Balance (col O), B14 → Rail Variance or GL Balance (col O)
  - But adjust based on what makes financial sense after reading the data.

## Step 3: Write a Python script to build the workbook

Use `openpyxl` to create the workbook. Key requirements:

1. **Do not modify any source files.**
2. Parse the CSV files with the `csv` module or `pandas`. Parse JSON with `json`.
3. For each detail sheet:
   - Row 1-5: headers/labels (program name, column headers for months, etc.)
   - Row 6+: line item data rows from the CSV
   - After line items: `Month Totals`, `Ending Balance`, `Variance`, `GL Balance` control rows with appropriate formulas or values
   - Column A should contain row labels; columns B onward should contain monthly numeric data
   - Column O should contain totals or the final relevant column
4. For the Transit Summary sheet:
   - Use Excel formula references (strings like `="='Bus Program #4310'!O{row}"`) for B7, B8, B9, B12, B13, B14
   - B16 must be the formula `=B9+B14`
5. All numeric cells must contain Python `int` or `float` values, never strings that look like numbers.
6. Save to `/root/MetroLink_Pass_Liability_4-25.xlsx`.

## Step 4: Validate the output

After creating the file:
1. Re-open it with openpyxl and verify:
   - Exactly 3 sheets with exact names in the correct order
   - Line items start at row 6 in detail sheets
   - Control rows exist with correct labels
   - Summary cells B7, B8, B9, B12, B13, B14 contain formula strings referencing column O of the detail tabs
   - B16 contains `=B9+B14`
   - Numeric cells are numeric (use `isinstance(cell.value, (int, float))`)
2. Print the sheet names, key cell values, and types for confirmation.

## Important notes
- The column layout and number of months must come from the actual CSV data — do not assume. Inspect first.
- The GL Balance values should come from `fare_liability_balances.json` or the ledger control notes.
- If the CSV has a beginning balance row or similar, place it appropriately before row 6 line items or as the first line item at row 6.
- Match the "Harbor reconciliation" pattern: this is a liability rollforward where you track beginning balance, additions, redemptions/usage, and arrive at an ending balance, then compare to GL.

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