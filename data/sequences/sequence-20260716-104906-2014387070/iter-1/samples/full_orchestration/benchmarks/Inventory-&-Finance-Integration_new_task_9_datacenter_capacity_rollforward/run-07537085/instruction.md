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

Read every file carefully. Understand:
- What months are covered (expect Jan–Jun or similar H1 2025 range)
- What line items exist for compute and storage
- What the ledger balances JSON contains (likely beginning/ending/GL balances)
- The operational context from the .txt files

### 2. Understand the "Harbor reconciliation" layout pattern

The reference layout is a roll-forward schedule. Each detail sheet has:
- **Row 1–5**: Headers. Column A = labels, Columns B–O (or similar) = months (likely Jan through some end month, with column O being the final/total column).
- **Row 6 onward**: Line items (individual capacity entries / transactions)
- After line items, four control rows in this exact order:
  - `Month Totals` — SUM of the line-item rows above, per column
  - `Ending Balance` — Beginning balance + Month Totals (running cumulative)
  - `Variance` — Ending Balance minus GL Balance
  - `GL Balance` — From the ledger balances JSON
- Column A has the row labels; columns B through O have monthly values.
- The first data column might be a "Beginning Balance" column, or months start at B.

### 3. Build the two detail sheets

Write a Python script using `openpyxl` to:

**For each detail sheet (`Compute Pool #8100` and `Storage Pool #8200`):**

a. Parse the corresponding CSV. Determine the column structure (months as columns). Place headers in rows 1-5 as appropriate (sheet title, column headers for months, etc.).

b. Line items start at **row 6**. Each row from the CSV becomes a line item row.

c. After all line items, add the four control rows:
- **Month Totals**: For each month column, `=SUM(cell_range)` summing all line-item cells in that column.
- **Ending Balance**: For each month column, this should be the beginning balance (from JSON) plus the cumulative month totals. If column B is the first month, `Ending Balance B = Beginning_Balance + Month_Totals_B`. For subsequent columns, `Ending Balance C = Ending Balance B + Month Totals C` (i.e., a running roll-forward). Alternatively, if the CSV already has a "Beginning Balance" row, adapt accordingly. Inspect the data to determine the exact formula pattern.
- **Variance**: `= Ending Balance - GL Balance` for each column (or just for column O / the final column).
- **GL Balance**: Hard-coded numeric values from the JSON.

d. **All numeric values must be stored as numbers, not strings.** When reading CSV, convert numeric fields with `float()` or `int()`. Watch for currency symbols, commas, parentheses (negatives) — strip them before conversion.

e. **Column O** is critical — it must contain the final period or total values that the Summary sheet references.

### 4. Build the Capacity Summary sheet

This sheet summarizes both detail sheets. The layout must have:
- Rows referencing Compute Pool #8100 results (rows 7-9 area) and Storage Pool #8200 results (rows 12-14 area)
- Specific cell formulas:
  - **B7** = `='Compute Pool #8100'!O<Month_Totals_row>` (links to Month Totals in column O of compute sheet)
  - **B8** = `='Compute Pool #8100'!O<Ending_Balance_row>`
  - **B9** = `='Compute Pool #8100'!O<Variance_row>`
  - **B12** = `='Storage Pool #8200'!O<Month_Totals_row>`
  - **B13** = `='Storage Pool #8200'!O<Ending_Balance_row>`
  - **B14** = `='Storage Pool #8200'!O<Variance_row>`
  - **B16** = `=B9+B14` (combined variance)

  Adjust the row references to match the actual control-row positions in each detail sheet.

### 5. Important cautions

- **Do NOT use `_xlfn.` prefixed function names.** Use plain `SUM`, `PERCENTILE`, etc. The cross-task feedback warns about `#NAME?` errors from function name issues — stick to basic functions (`SUM`, simple arithmetic) which are safe.
- **Do NOT modify any source files.**
- **Ensure sheet order** is exactly: `Capacity Summary`, `Compute Pool #8100`, `Storage Pool #8200`. In openpyxl, create them in order or use `wb.move_sheet()` to reorder.
- **Verify numeric types**: After writing, re-open the workbook and spot-check that cells contain numbers (not strings that look like numbers).

### 6. Validation

After creating the workbook, run these checks:
```python
import openpyxl
wb = openpyxl.load_workbook('/root/Nimbus_Capacity_Reconciliation_4-25.xlsx')
print('Sheet names:', wb.sheetnames)  # Must be exactly ['Capacity Summary', 'Compute Pool #8100', 'Storage Pool #8200']

# Check Capacity Summary formulas
ws = wb['Capacity Summary']
for cell in ['B7','B8','B9','B12','B13','B14','B16']:
    print(f'{cell}: {ws[cell].value}')
# B7,B8,B9 should be formulas referencing 'Compute Pool #8100'!O...
# B12,B13,B14 should be formulas referencing 'Storage Pool #8200'!O...
# B16 should be '=B9+B14'

# Check detail sheets
for name in ['Compute Pool #8100', 'Storage Pool #8200']:
    ws = wb[name]
    print(f'\n--- {name} ---')
    # Print rows around line items and control rows
    for row in ws.iter_rows(min_row=1, max_col=16, values_only=False):
        vals = [(c.coordinate, c.value) for c in row if c.value is not None]
        if vals:
            print(vals)
    # Verify row 6 is first line item
    print(f'Row 6, col A: {ws.cell(6,1).value}')
    # Check that numeric cells are actually numeric
    sample = ws.cell(6,2).value
    print(f'Row 6, col B value={sample}, type={type(sample)}')
```

Fix any issues found. The workbook must be saved at exactly `/root/Nimbus_Capacity_Reconciliation_4-25.xlsx`.

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