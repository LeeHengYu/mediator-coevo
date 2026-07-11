# Task Instruction

You are the Cloud Capacity Operations Lead at Nimbus Compute. Build an Excel workbook at `/root/Nimbus_Capacity_Reconciliation_4-25.xlsx` by following these steps precisely.

## Step 0 — Inspect all input files

```bash
cat /root/compute_capacity_schedule_input.csv
cat /root/storage_capacity_schedule_input.csv
cat /root/capacity_ledger_balances.json
cat /root/Nimbus_Compute_Reservation_Register_Q1Q2_2025.txt
cat /root/Nimbus_Storage_Commitment_Register_Q1Q2_2025.txt
cat /root/nimbus_platform_ledger_notes_apr25.txt
```

Read every file carefully before writing any code. Understand:
- What columns the CSVs have (likely: description/line-item, then monthly columns Jan–Jun or similar)
- What the JSON ledger contains (likely GL balances for compute and storage pools)
- What contextual info the .txt files provide

## Step 1 — Understand the structure contract

The output workbook must have exactly 3 sheets in this order:
1. `Capacity Summary`
2. `Compute Pool #8100`
3. `Storage Pool #8200`

The two detail sheets (`Compute Pool #8100` and `Storage Pool #8200`) follow this layout:
- Row 1: sheet title
- Row 2–4: headers (column A = labels, columns B–O or similar = months)
- Row 5: possibly a header row or blank
- **Row 6 onward: line items** (data from the CSV inputs)
- After line items, these **control rows** in order:
  - `Month Totals` — SUM of all line-item cells in that column
  - `Ending Balance` — computed (likely Opening Balance + Month Totals, or a running balance)
  - `Variance` — difference between Ending Balance and GL Balance
  - `GL Balance` — from the JSON ledger

The `Capacity Summary` sheet:
- Has a compact layout summarizing both pools
- Rows 7–9 relate to Compute Pool #8100 (B7 = label or value, B8, B9)
- Rows 12–14 relate to Storage Pool #8200 (B12, B13, B14)
- Row 16: combined/total
- **B7** links to column O of `Compute Pool #8100` (likely Ending Balance in last month column)
- **B8** links to column O of `Compute Pool #8100` (likely GL Balance)
- **B9** links to column O of `Compute Pool #8100` (likely Variance)
- **B12** links to column O of `Storage Pool #8200` (Ending Balance)
- **B13** links to column O of `Storage Pool #8200` (GL Balance)
- **B14** links to column O of `Storage Pool #8200` (Variance)
- **B16 = B9 + B14** (combined variance)

IMPORTANT: The exact mapping of B7/B8/B9 to which control row (Ending Balance, GL Balance, Variance) depends on what makes logical sense. Inspect the data and adapt. The key constraint is that B16 = B9 + B14, and these cells reference column O of the detail tabs.

## Step 2 — Write a Python script to build the workbook

Use `openpyxl` to create the workbook. Key rules:
- All numeric values must be stored as numbers (int or float), NOT strings.
- Use Excel formulas where specified (summary sheet references to detail sheets, SUM formulas for Month Totals, etc.).
- Do NOT modify any source files.
- Column A = row labels/descriptions. Columns B through O (or however many months) = monthly data.
- Line items start at row 6 in the detail sheets.

For the detail sheets:
1. Row 1: Sheet title (e.g., "Compute Pool #8100 - Capacity Roll-Forward")
2. Rows 2-5: Headers — inspect the CSV to determine column headers (months). Place them starting at row 5 or wherever makes sense, but ensure line items start at row 6.
3. Row 6+: Line items from the CSV. Each row = one line item. Column A = description, columns B onward = monthly values.
4. After the last line item, insert control rows:
   - `Month Totals`: Each cell = SUM of that column's line-item cells (row 6 to last line-item row)
   - `Ending Balance`: Depends on context — likely an opening balance (from JSON or first-row logic) plus cumulative Month Totals, or a formula. Inspect the data to determine the right formula.
   - `Variance`: `= Ending Balance - GL Balance` for each month
   - `GL Balance`: Values from the JSON ledger, placed as numbers

For the summary sheet:
1. Set up labels in column A for rows 6-16.
2. B7, B8, B9 reference column O of `Compute Pool #8100` control rows (use formulas like `='Compute Pool #8100'!O{row}`).
3. B12, B13, B14 reference column O of `Storage Pool #8200` control rows.
4. B16 formula: `=B9+B14`

## Step 3 — Validate

After creating the workbook, run a validation script:
```python
import openpyxl
wb = openpyxl.load_workbook('/root/Nimbus_Capacity_Reconciliation_4-25.xlsx')
print('Sheet names:', wb.sheetnames)
assert wb.sheetnames == ['Capacity Summary', 'Compute Pool #8100', 'Storage Pool #8200']

# Check detail sheets
for name in ['Compute Pool #8100', 'Storage Pool #8200']:
    ws = wb[name]
    # Verify line items start at row 6
    print(f'{name} A6:', ws['A6'].value)
    # Find control rows
    for row in ws.iter_rows(min_col=1, max_col=1):
        for cell in row:
            if cell.value and isinstance(cell.value, str) and cell.value.strip() in ['Month Totals', 'Ending Balance', 'Variance', 'GL Balance']:
                print(f'{name} control row: {cell.value} at row {cell.row}')

# Check summary sheet
ws = wb['Capacity Summary']
for r in [7,8,9,12,13,14,16]:
    cell = ws.cell(row=r, column=2)
    print(f'Summary B{r}: value={cell.value}, type={type(cell.value)}')

# Verify B16 formula
print('B16 raw:', ws['B16'].value)
```

Check that:
- All three sheets exist in the correct order
- Line items start at row 6
- Control rows exist with correct labels
- Summary formulas reference the detail sheets
- B16 = B9 + B14
- All numeric cells contain numbers, not strings

## Critical Notes
- Adapt the exact row numbers for control rows based on how many line items exist in each CSV.
- The column count (B through O = 14 columns) suggests 14 months or periods. Inspect the CSV headers to confirm.
- If the JSON contains opening balances, use them appropriately (e.g., as a row before line items or as the basis for Ending Balance calculation).
- If any step is ambiguous, make conservative assumptions consistent with a standard roll-forward schedule (Opening Balance + Period Activity = Ending Balance).

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