# Task Instruction

You are the Cloud Capacity Operations Lead at Nimbus Compute. Build an Excel workbook at `/root/Nimbus_Capacity_Reconciliation_4-25.xlsx` by following these steps precisely.

## Step 1: Inspect all input files

```bash
cat /root/compute_capacity_schedule_input.csv
cat /root/storage_capacity_schedule_input.csv
cat /root/capacity_ledger_balances.json
cat /root/Nimbus_Compute_Reservation_Register_Q1Q2_2025.txt
cat /root/Nimbus_Storage_Commitment_Register_Q1Q2_2025.txt
cat /root/nimbus_platform_ledger_notes_apr25.txt
```

Read every file carefully before writing any code. Understand:
- What months/columns are covered (expect columns A through O, where O is likely a totals or ending column)
- What line items exist for compute and storage
- What the ledger balances JSON contains (GL balances, beginning balances, etc.)
- The operational context documents for any additional data points

## Step 2: Understand the required structure

The workbook must have exactly 3 sheets in this order:
1. `Capacity Summary`
2. `Compute Pool #8100`
3. `Storage Pool #8200`

### Detail sheets (`Compute Pool #8100` and `Storage Pool #8200`):
- Follow a roll-forward schedule structure similar to a harbor reconciliation
- Row 1-5: Headers (row 1 = title, row 2-4 can be sub-headers/blank, row 5 = column headers with month names)
- **Line items start at row 6**
- Column A = line item labels
- Columns B through N = monthly data (or however many months the CSV covers)
- Column O = totals or ending values
- After line items, include these **control rows** (in order):
  - `Month Totals` — SUM of all line item values for each month column
  - `Ending Balance` — Beginning Balance + Month Totals (rolling forward)
  - `Variance` — Ending Balance minus GL Balance
  - `GL Balance` — from the ledger balances JSON
- The `Beginning Balance` row should appear before the line items or as the first data concept

### `Capacity Summary` sheet:
- Must have summary formulas that link to column O of the two detail tabs
- Specific cell requirements:
  - B7 links to a value from `Compute Pool #8100` column O
  - B8 links to a value from `Compute Pool #8100` column O  
  - B9 links to a value from `Compute Pool #8100` column O
  - B12 links to a value from `Storage Pool #8200` column O
  - B13 links to a value from `Storage Pool #8200` column O
  - B14 links to a value from `Storage Pool #8200` column O
  - **B16 must be the formula `=B9+B14`** (sum of the two pool ending/total values)
- Determine which control-row values from column O map to B7-B9 and B12-B14 by examining the data. Likely mapping:
  - B7 = Compute Beginning Balance (or Month Totals) from col O
  - B8 = Compute Month Totals (or Ending Balance) from col O  
  - B9 = Compute Ending Balance (or Variance) from col O
  - B12 = Storage Beginning Balance (or Month Totals) from col O
  - B13 = Storage Month Totals (or Ending Balance) from col O
  - B14 = Storage Ending Balance (or Variance) from col O
  - B16 = B9 + B14
- Look at the data to determine the exact mapping. The pattern B7/B8/B9 for compute and B12/B13/B14 for storage with B16=B9+B14 suggests rows 7-9 are one block, rows 12-14 are another, with labels in column A.

## Step 3: Write a Python script to build the workbook

Use `openpyxl` to create the workbook. Key requirements:

1. **All numeric values must be stored as numbers** (int or float), NOT as strings. When reading CSV, convert numeric fields explicitly with `float()` or `int()`.
2. **Use Excel formulas** where specified:
   - `Month Totals` rows: use SUM formulas over the line-item range for each column
   - `Ending Balance` rows: formula = Beginning Balance cell + Month Totals cell
   - `Variance` rows: formula = Ending Balance - GL Balance
   - Summary sheet B7-B14: use cross-sheet references like `='Compute Pool #8100'!O<row>`
   - B16: use formula `=B9+B14`
3. **Do not modify source files.**
4. **Sheet order matters** — create sheets in the exact order specified.
5. Parse the CSV files to determine line item names and monthly values.
6. Parse the JSON for GL balances and beginning balances.

## Step 4: Validate the output

After creating the workbook:
```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('/root/Nimbus_Capacity_Reconciliation_4-25.xlsx')
print('Sheet names:', wb.sheetnames)
for name in wb.sheetnames:
    ws = wb[name]
    print(f'\n=== {name} ===')
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column, values_only=False):
        print([(c.coordinate, c.value) for c in row])
"
```

Verify:
- Exactly 3 sheets in correct order: `Capacity Summary`, `Compute Pool #8100`, `Storage Pool #8200`
- Line items start at row 6 on detail sheets
- Control rows exist with correct labels: `Month Totals`, `Ending Balance`, `Variance`, `GL Balance`
- Summary sheet B7, B8, B9 reference `Compute Pool #8100` column O
- Summary sheet B12, B13, B14 reference `Storage Pool #8200` column O
- B16 formula is `=B9+B14`
- All numeric cells contain actual numbers, not strings
- The file exists at `/root/Nimbus_Capacity_Reconciliation_4-25.xlsx`

## Critical Notes
- From cross-task feedback: Column ordering matters enormously. Double-check every column header and row label matches exactly what the specification requires.
- The control row labels must be exactly: `Month Totals`, `Ending Balance`, `Variance`, `GL Balance` (check exact spelling/capitalization)
- Keep numeric values numeric — this was explicitly called out in the task.
- B16 MUST be the formula `=B9+B14`, not a hardcoded value.

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