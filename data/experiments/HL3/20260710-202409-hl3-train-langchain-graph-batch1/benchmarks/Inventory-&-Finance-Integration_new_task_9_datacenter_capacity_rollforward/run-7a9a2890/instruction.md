# Task Instruction

## Task: Build Nimbus Capacity Reconciliation Workbook

You must create an Excel workbook at `/root/Nimbus_Capacity_Reconciliation_4-25.xlsx` with exactly three sheets in this order: `Capacity Summary`, `Compute Pool #8100`, `Storage Pool #8200`.

### Step-by-step instructions:

#### 1. Inspect all input files first

```bash
cat /root/compute_capacity_schedule_input.csv
cat /root/storage_capacity_schedule_input.csv
cat /root/capacity_ledger_balances.json
cat /root/Nimbus_Compute_Reservation_Register_Q1Q2_2025.txt
cat /root/Nimbus_Storage_Commitment_Register_Q1Q2_2025.txt
cat /root/nimbus_platform_ledger_notes_apr25.txt
```

Read every file carefully. Understand the data structure, column names, monthly values, and any ledger balances before writing any code.

#### 2. Understand the required structure

The workbook follows a "Harbor reconciliation" pattern:

**Detail sheets (`Compute Pool #8100` and `Storage Pool #8200`):**
- Row 1: Sheet title/header
- Row 2-5: Header rows (column headers, etc.) — adapt based on what makes sense for the data
- **Row 6 onward: Line items** — each row is a line item from the CSV input. The columns should include an item identifier/description in column A, then monthly values in columns B through N (or similar), with **column O containing a total/summary** for that line item (e.g., SUM of monthly values).
- After the line items, include these **control rows** in order:
  - `Month Totals` — sum of all line items for each month column
  - `Ending Balance` — the ending balance (derived from beginning balance + month totals, or as specified by the data)
  - `Variance` — difference between Ending Balance and GL Balance
  - `GL Balance` — from `capacity_ledger_balances.json`

**`Capacity Summary` sheet:**
- This sheet summarizes the two detail sheets.
- Cell references (these are critical and will be verified):
  - `B7` = links to `Compute Pool #8100` column O ending balance or relevant summary value
  - `B8` = links to `Compute Pool #8100` column O relevant value
  - `B9` = links to `Compute Pool #8100` column O relevant value
  - `B12` = links to `Storage Pool #8200` column O relevant value
  - `B13` = links to `Storage Pool #8200` column O relevant value
  - `B14` = links to `Storage Pool #8200` column O relevant value
  - **`B16` must be a formula: `=B9+B14`** (combined total of compute and storage)
- The exact mapping of B7/B8/B9 to compute detail rows and B12/B13/B14 to storage detail rows should correspond to the key control rows (e.g., Ending Balance, Variance, GL Balance or Month Totals, Ending Balance, GL Balance — determine from the data context).

#### 3. Build the workbook using openpyxl

Write a Python script that:

1. Reads the CSV files with `csv` module or `openpyxl` (do NOT use pandas to avoid dependency issues — check if pandas is available first; if not, use csv + openpyxl).
2. Reads the JSON file for ledger balances.
3. Creates the workbook with exactly three sheets in the specified order.
4. Populates the detail sheets with:
   - Line items starting at row 6
   - Monthly columns (determine from CSV headers)
   - Column O containing row totals (SUM formulas across monthly columns)
   - Control rows after line items: `Month Totals`, `Ending Balance`, `Variance`, `GL Balance`
5. Populates the `Capacity Summary` sheet with:
   - Labels and cross-sheet formula references
   - B7, B8, B9 referencing `'Compute Pool #8100'!O{row}` for the appropriate control rows
   - B12, B13, B14 referencing `'Storage Pool #8200'!O{row}` for the appropriate control rows
   - **B16 = `=B9+B14`** (must be an Excel formula, not a static value)
6. Ensures all numeric values are stored as numbers (int or float), NOT as strings.
7. Saves to `/root/Nimbus_Capacity_Reconciliation_4-25.xlsx`.

#### 4. Validate the output

After creating the workbook, write a validation script that:
- Opens the workbook
- Confirms exactly 3 sheets in the correct order
- Confirms line items start at row 6 on detail sheets
- Confirms control row labels exist (`Month Totals`, `Ending Balance`, `Variance`, `GL Balance`)
- Confirms B16 on `Capacity Summary` is a formula containing `B9+B14`
- Confirms B7, B8, B9, B12, B13, B14 on `Capacity Summary` contain formulas referencing the detail sheets' column O
- Confirms numeric cells contain numbers, not strings
- Prints all values and formulas for inspection

#### 5. Important constraints
- Do NOT modify any source files.
- All numeric values must be numeric type in Excel (not text).
- The final file must be `.xlsx` format at the exact path specified.
- Use `round()` to 2 decimal places for any calculated floating-point values to avoid precision drift.
- Column O on detail sheets must contain the values that the summary sheet references.

#### 6. Debugging approach
- If the CSV structure is unclear, print the first few rows and all headers before processing.
- If the JSON structure is unclear, print it formatted before processing.
- Map out which rows will be which control rows BEFORE writing the workbook, and print the mapping.
- After creation, re-read and print key cells to verify correctness.

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