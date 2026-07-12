# Task Instruction

## Task: Build MetroLink_Pass_Liability_4-25.xlsx

You are building an Excel workbook that reconciles transit pass liability for MetroLink Transit Authority. Follow these steps precisely.

### Step 0 — Inspect all input files

```bash
cat /root/bus_pass_schedule_input.csv
cat /root/rail_pass_schedule_input.csv
python3 -c "import json, pprint; pprint.pprint(json.load(open('/root/fare_liability_balances.json')))"
cat /root/MetroLink_Bus_Pass_Issuance_Notes_Q1Q2_2025.txt
cat /root/MetroLink_Rail_Pass_Issuance_Notes_Q1Q2_2025.txt
cat /root/metrolink_fare_ledger_control_notes_apr25.txt
```

Read every file carefully. Understand the column structure of the CSVs (months as columns), the JSON balances (beginning/ending/GL balances per program), and any operational context from the text files.

### Step 1 — Understand the required workbook structure

The workbook must have exactly 3 sheets in this order:
1. `Transit Summary`
2. `Bus Program #4310`
3. `Rail Program #4320`

**Detail sheets (Bus Program #4310, Rail Program #4320):**
- Row 1: Title/header row
- Row 2–5: Header area (column headers in row 5, with months in columns B through O where column O is the total/sum column, or however the CSV maps — inspect the CSV to determine exact month layout)
- Row 6 onward: Line items from the CSV (each row of data from the CSV becomes a row starting at row 6)
- After line items, control rows in order:
  - `Month Totals` — sum of all line-item values for each month column
  - `Ending Balance` — computed from Beginning Balance minus Month Totals (or as indicated by the data/context)
  - `Variance` — difference between Ending Balance and GL Balance
  - `GL Balance` — from the JSON file
- Column A has labels; columns B–O (or as many month columns as exist) have numeric values.
- The last column (column O if months span B–N with O as total, or adjust based on CSV inspection) should contain row totals or year-to-date totals.

**Transit Summary sheet:**
- This is a summary that references the detail sheets.
- Structure (based on the Harbor reconciliation pattern):
  - B7 = Bus Program beginning balance (from Bus detail tab column O or relevant total)
  - B8 = Bus Program month totals (links to Bus detail tab Month Totals, column O)
  - B9 = Bus Program ending balance (links to Bus detail tab Ending Balance, column O)
  - B12 = Rail Program beginning balance (from Rail detail tab column O)
  - B13 = Rail Program month totals (links to Rail detail tab Month Totals, column O)
  - B14 = Rail Program ending balance (links to Rail detail tab Ending Balance, column O)
  - B16 = B9 + B14 (combined ending balance)
- Add appropriate labels in column A for these rows.

### Step 2 — Build the workbook with openpyxl

Write a Python script that:
1. Reads bus_pass_schedule_input.csv and rail_pass_schedule_input.csv using the `csv` module.
2. Reads fare_liability_balances.json.
3. Creates the workbook with openpyxl.
4. For each detail sheet:
   a. Write a title in row 1.
   b. Write column headers in row 5 (matching CSV headers).
   c. Write line-item data starting at row 6, converting all numeric strings to floats/ints.
   d. After the last line item, write control rows: Beginning Balance, Month Totals, Ending Balance, Variance, GL Balance.
   e. Beginning Balance and GL Balance come from the JSON.
   f. Month Totals = sum of line-item cells in that column (compute in Python, write as numbers).
   g. Ending Balance = Beginning Balance - Month Totals (or + depending on the sign convention — check the data).
   h. Variance = Ending Balance - GL Balance.
   i. Column O (the last data column) should have the row total across all months for each line item, and corresponding totals for control rows.
5. For the Transit Summary sheet, write labels and values that reference the computed values from the detail sheets. Write actual numeric values (computed in Python), not Excel formulas, to ensure the verifier sees numbers.
6. B16 MUST equal B9 + B14.
7. Ensure all numeric cells contain Python numbers (int or float), not strings.
8. Save to `/root/MetroLink_Pass_Liability_4-25.xlsx`.

### Step 3 — Validate

After creating the workbook, verify it:
```python
import openpyxl
wb = openpyxl.load_workbook('/root/MetroLink_Pass_Liability_4-25.xlsx')
print('Sheets:', wb.sheetnames)
for name in wb.sheetnames:
    ws = wb[name]
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column, values_only=False):
        print(name, [(c.coordinate, c.value, type(c.value).__name__) for c in row])
```

Check:
- Exactly 3 sheets in correct order
- Line items start at row 6
- Control rows present with correct labels
- B16 on Transit Summary = B9 + B14
- All numeric values are numeric types
- Column O on detail sheets has totals
- Summary cells B7/B8/B9/B12/B13/B14 match the detail sheet column O values

### Critical Notes
- Do NOT modify any source files.
- Adapt the exact row/column mapping based on what you find in the CSVs. The CSVs may have months as columns. Map them faithfully.
- The sign convention for Beginning Balance, issuances (additions to liability), redemptions (reductions), and Ending Balance should follow standard liability accounting: Beginning Balance + New Issuances - Redemptions = Ending Balance. Check the data to confirm.
- If the JSON has separate entries for bus (4310) and rail (4320), map them accordingly.
- Write computed values, not Excel formulas, to ensure the verifier reads actual numbers.

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