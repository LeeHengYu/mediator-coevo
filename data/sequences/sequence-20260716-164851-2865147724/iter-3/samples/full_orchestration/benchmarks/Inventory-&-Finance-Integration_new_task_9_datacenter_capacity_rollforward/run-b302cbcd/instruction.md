# Task Instruction

Build the Excel workbook `/root/Nimbus_Capacity_Reconciliation_4-25.xlsx` with exactly three sheets in order: `Capacity Summary`, `Compute Pool #8100`, `Storage Pool #8200`.

## Step-by-step plan

### 1. Inspect all input files

```bash
cat /root/compute_capacity_schedule_input.csv
cat /root/storage_capacity_schedule_input.csv
cat /root/capacity_ledger_balances.json
cat /root/Nimbus_Compute_Reservation_Register_Q1Q2_2025.txt
cat /root/Nimbus_Storage_Commitment_Register_Q1Q2_2025.txt
cat /root/nimbus_platform_ledger_notes_apr25.txt
```

Read every file carefully before writing any code. Understand:
- What months are covered (likely Jan–Jun or similar)
- What line items exist for compute and storage
- What the ledger balances JSON contains (likely GL balances, beginning balances, etc.)
- The column structure: likely Column A = line item labels, Columns B–O = months or categories, with Column O being a total/summary column

### 2. Understand the reference Harbor reconciliation layout

The task references a "Harbor reconciliation task" layout. The key structural rules are:

**Detail sheets (Compute Pool #8100, Storage Pool #8200):**
- Row 1–5: Headers (title, column headers for months, etc.)
- Row 6 onward: Line items from the CSV input data
- After line items, control rows in this order:
  - `Month Totals` — SUM of the line item rows above for each month column
  - `Ending Balance` — Beginning Balance + Month Totals (running balance)
  - `Variance` — Ending Balance minus GL Balance
  - `GL Balance` — from the ledger JSON
- Column A = labels, Columns B through N = individual months, Column O = total/summary

**Capacity Summary sheet:**
- Row 6: label for compute beginning balance → B6 links somewhere
- B7 = links to Compute Pool #8100 column O Month Totals
- B8 = links to Compute Pool #8100 column O Ending Balance  
- B9 = links to Compute Pool #8100 column O Variance (or GL Balance — determine from data)
- B12 = links to Storage Pool #8200 column O Month Totals
- B13 = links to Storage Pool #8200 column O Ending Balance
- B14 = links to Storage Pool #8200 column O Variance (or GL Balance)
- B16 = `=B9+B14` (combined total)

The exact mapping of B7/B8/B9/B12/B13/B14 to detail sheet control rows needs to be inferred from the data. The instruction says B7/B8/B9 link to column O of the compute detail tab and B12/B13/B14 link to column O of the storage detail tab. B16 = B9+B14.

### 3. Write a Python script to build the workbook

Use `openpyxl` to create the workbook. The script should:

1. Parse the two CSV files with `csv.reader` or `pandas`.
2. Parse the JSON ledger balances.
3. Determine the month columns from the CSV headers.
4. Build the two detail sheets:
   - Row 1: Sheet title (e.g., "Compute Pool #8100")
   - Row 2–4: Can be blank or contain sub-headers
   - Row 5: Column headers (Month names across B–N, "Total" in O)
   - Row 6+: Line items from CSV, with numeric values stored as numbers (float/int), NOT strings
   - After last line item row: `Month Totals` row with SUM formulas
   - `Ending Balance` row: Beginning balance (from JSON) + Month Totals
   - `Variance` row: Ending Balance - GL Balance
   - `GL Balance` row: values from JSON
   - Column O for each line item row: SUM(B:N) for that row
   - Column O for control rows: corresponding SUM or formula

5. Build the Capacity Summary sheet:
   - B7 formula: `='Compute Pool #8100'!O{month_totals_row}`
   - B8 formula: `='Compute Pool #8100'!O{ending_balance_row}`
   - B9 formula: `='Compute Pool #8100'!O{variance_row}` (or whichever control row)
   - B12–B14: same pattern for Storage Pool #8200
   - B16: `=B9+B14`

**Critical details:**
- All numeric values must be stored as numbers, not strings. When reading CSV, convert to float/int.
- Use Excel formulas (not hardcoded values) for SUM, cross-sheet references, and B16.
- Sheet names must be exactly: `Capacity Summary`, `Compute Pool #8100`, `Storage Pool #8200`
- Sheet order must match exactly.

### 4. Validate the output

After creating the workbook, run a validation script:
```python
import openpyxl
wb = openpyxl.load_workbook('/root/Nimbus_Capacity_Reconciliation_4-25.xlsx')
print('Sheet names:', wb.sheetnames)
assert wb.sheetnames == ['Capacity Summary', 'Compute Pool #8100', 'Storage Pool #8200']

# Check detail sheets have control rows
for sname in ['Compute Pool #8100', 'Storage Pool #8200']:
    ws = wb[sname]
    labels = [ws.cell(row=r, column=1).value for r in range(1, ws.max_row+1)]
    print(f'{sname} labels: {labels}')
    assert 'Month Totals' in labels
    assert 'Ending Balance' in labels
    assert 'Variance' in labels
    assert 'GL Balance' in labels

# Check summary formulas
ws = wb['Capacity Summary']
for cell in ['B7','B8','B9','B12','B13','B14','B16']:
    print(f'{cell}: {ws[cell].value}')
# B7-B9 should be formulas referencing Compute Pool #8100
# B12-B14 should be formulas referencing Storage Pool #8200
# B16 should be =B9+B14
assert ws['B16'].value == '=B9+B14' or ws['B16'].value == '=B9+B14'

# Check numeric values in detail sheets
for sname in ['Compute Pool #8100', 'Storage Pool #8200']:
    ws = wb[sname]
    for r in range(6, ws.max_row+1):
        for c in range(2, ws.max_column+1):
            v = ws.cell(row=r, column=c).value
            if v is not None and not isinstance(v, str):
                assert isinstance(v, (int, float)), f'Non-numeric at {r},{c}: {type(v)}'
```

### 5. Key warnings from feedback
- Ensure all numeric CSV values are converted to actual numbers, not left as strings.
- Do not modify any source files.
- Match exact sheet names including the `#` symbol.
- Use proper thousands-separator formatting if any display formatting is needed, but the core requirement is numeric cell values.
- Adapt the row layout based on what you actually find in the CSV/JSON files — do NOT assume a fixed number of line items. Count them from the data.
- The line items start at row 6 means the FIRST data line item is in row 6. Headers/titles occupy rows 1-5.

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