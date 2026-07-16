# Task Instruction

Build the Excel workbook `/root/MetroLink_Pass_Liability_4-25.xlsx` with exactly three sheets in order: `Transit Summary`, `Bus Program #4310`, `Rail Program #4320`.

## Step-by-step plan

### 1. Inspect all input files

```bash
cat /root/bus_pass_schedule_input.csv
cat /root/rail_pass_schedule_input.csv
cat /root/fare_liability_balances.json
cat /root/MetroLink_Bus_Pass_Issuance_Notes_Q1Q2_2025.txt
cat /root/MetroLink_Rail_Pass_Issuance_Notes_Q1Q2_2025.txt
cat /root/metrolink_fare_ledger_control_notes_apr25.txt
```

Read every file carefully before writing any code. Understand:
- What columns exist in each CSV (likely months as columns, with line-item rows)
- What balances are in the JSON (likely beginning/ending/GL balances for each program)
- What operational context the text files provide

### 2. Inspect any reference/verifier files

```bash
find /root -name '*.py' | head -20
cat /root/test_output.py 2>/dev/null || true
find /root -name 'test_*' -o -name '*test*' | head -20
```

Read the verifier/test script completely to understand exactly what is checked: sheet names, cell addresses, formula patterns, numeric values, row structure, etc.

### 3. Understand the required structure

Based on the task description and the "Harbor reconciliation" reference pattern:

**Detail sheets (`Bus Program #4310` and `Rail Program #4320`):**
- Row 1: Title/header
- Row 2-4: Possibly sub-headers or blank
- Row 5: Column headers (A = line item description, B through N = months Jan-Dec or similar, O = totals)
- Row 6 onward: Line item data rows from the CSV inputs
- After line items: control rows in this order:
  - `Month Totals` row (SUM of line items above, per column)
  - `Ending Balance` row
  - `Variance` row
  - `GL Balance` row
- Column O should contain annual/period totals (SUM across months for each row)

**`Transit Summary` sheet:**
- Links to the detail tabs
- B7 = reference to Bus Program #4310 column O Ending Balance (or similar)
- B8 = reference to Bus Program #4310 column O for another metric
- B9 = reference to Bus Program #4310 column O for another metric
- B12, B13, B14 = same pattern but for Rail Program #4320
- B16 = B9 + B14 (combined total)

### 4. Build the workbook with openpyxl

Write a Python script that:

a) Reads `bus_pass_schedule_input.csv` and `rail_pass_schedule_input.csv` using the csv module.
b) Reads `fare_liability_balances.json` using json module.
c) Creates the workbook with `openpyxl`.
d) Populates `Bus Program #4310` sheet:
   - Place headers in row 5 (or wherever the CSV headers map)
   - Place line item data starting at row 6
   - All numeric values stored as Python floats/ints, NOT strings
   - After the last line item row, add control rows: `Month Totals`, `Ending Balance`, `Variance`, `GL Balance`
   - `Month Totals` row: Excel SUM formulas summing the line-item rows for each month column
   - Column O for each line item: Excel SUM formula across months
   - `Ending Balance`, `GL Balance` from the JSON data
   - `Variance` = Ending Balance - GL Balance (as formula)
e) Populate `Rail Program #4320` sheet identically but with rail data.
f) Populate `Transit Summary` sheet:
   - B7, B8, B9 reference column O cells from `Bus Program #4310`
   - B12, B13, B14 reference column O cells from `Rail Program #4320`
   - B16 formula: `=B9+B14`
   - Use Excel cross-sheet references like `='Bus Program #4310'!O<row>`
g) Save to `/root/MetroLink_Pass_Liability_4-25.xlsx`

### 5. Critical rules
- All numeric cell values must be numeric types, not strings.
- Do NOT modify any source files.
- Sheet order must be exactly: `Transit Summary`, `Bus Program #4310`, `Rail Program #4320`.
- Sheet names must match exactly (including spaces, `#`, and numbers).
- The summary cell B16 MUST use the formula `=B9+B14` (not a hardcoded value).
- Cross-sheet references in B7/B8/B9/B12/B13/B14 must be Excel formulas referencing column O of the detail tabs.
- Control row labels must match exactly: `Month Totals`, `Ending Balance`, `Variance`, `GL Balance`.

### 6. Validate

After creating the workbook:
```python
# Re-open and verify
import openpyxl
wb = openpyxl.load_workbook('/root/MetroLink_Pass_Liability_4-25.xlsx')
print('Sheets:', wb.sheetnames)
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'\n--- {sn} ---')
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column, values_only=False):
        for cell in row:
            if cell.value is not None:
                print(f'  {cell.coordinate}: {repr(cell.value)} (type={type(cell.value).__name__})')
```

Then run the verifier:
```bash
cd /root && python test_output.py 2>&1 || true
```

If the verifier fails, read the error output carefully, identify what's wrong, fix it, and re-run. Pay special attention to:
- Exact cell addresses being checked
- Expected values vs actual values
- Formula format expectations
- Row numbers for control rows
- Whether the verifier uses data_only mode or checks formulas

### 7. Adapt to what you find

The exact mapping of CSV columns to sheet columns, which JSON fields map to which control rows, and which detail-sheet rows are referenced by B7/B8/B9/B12/B13/B14 depend on the actual file contents. After reading the files in step 1 and the verifier in step 2, adapt the plan accordingly. The CSV structure and JSON keys will tell you exactly what goes where. The verifier will tell you exactly what cells and values are expected.

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