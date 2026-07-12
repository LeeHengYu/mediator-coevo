# Task Instruction

You are the Fare Programs Analyst at MetroLink Transit Authority. Build an Excel workbook at `/root/MetroLink_Pass_Liability_4-25.xlsx` with exactly three sheets in this order: `Transit Summary`, `Bus Program #4310`, `Rail Program #4320`.

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

Read every file carefully before writing any code. Understand the column structure, month columns, line-item rows, and any balances/GL figures.

### 2. Inspect the verifier

Look for any test file that will check the output:
```bash
find / -name 'test_output*' -o -name 'test_*.py' -o -name 'test_*.js' 2>/dev/null | head -20
```
Read the verifier file(s) thoroughly. Identify every cell reference, expected value, sheet name, formula pattern, and structural check. This is the contract you must satisfy exactly.

### 3. Understand the required structure

Based on the task description and the "Harbor reconciliation" reference pattern:

**Detail sheets (`Bus Program #4310` and `Rail Program #4320`):**
- Row 1: Title/header row
- Row 2-5: Header area (column headers likely in row 5, with months in columns B through N or similar, and column O for totals)
- Row 6 onward: Line items from the CSV input (one row per line item)
- After line items, control rows in this exact order:
  - `Month Totals` — SUM of the line-item values above, per column
  - `Ending Balance` — derived from beginning balance + month totals (or as indicated by input data / fare_liability_balances.json)
  - `Variance` — difference between Ending Balance and GL Balance
  - `GL Balance` — from fare_liability_balances.json
- Column O should contain the total across all months for each row

**Summary sheet (`Transit Summary`):**
- B7 = reference to Bus detail tab column O Ending Balance
- B8 = reference to Bus detail tab column O Variance  
- B9 = reference to Bus detail tab column O GL Balance
- B12 = reference to Rail detail tab column O Ending Balance
- B13 = reference to Rail detail tab column O Variance
- B14 = reference to Rail detail tab column O GL Balance
- B16 = B9 + B14 (combined GL balance)

**IMPORTANT**: The exact mapping of B7/B8/B9/B12/B13/B14 to which control row on the detail tabs may differ — read the verifier to confirm the exact expected values or formula patterns. The verifier is the ground truth.

### 4. Write the Python script

Use `openpyxl` to create the workbook. Key rules:
- All numeric values must be stored as numbers (int or float), NOT strings.
- Use Excel formulas (e.g., `=SUM(...)`) where appropriate, especially for Month Totals, Ending Balance, Variance, and the summary sheet cross-references.
- For summary sheet formulas referencing detail tabs, use the exact sheet names with proper quoting: e.g., `='Bus Program #4310'!O<row>`.
- B16 must be the formula `=B9+B14`.
- Column A should contain row labels. Data columns start at B.
- Line items start at row 6 on detail sheets.
- Do NOT modify any source files.

### 5. Validate the output

After generating the workbook:
```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('/root/MetroLink_Pass_Liability_4-25.xlsx')
print('Sheets:', wb.sheetnames)
for name in wb.sheetnames:
    ws = wb[name]
    print(f'\n=== {name} ===')
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column, values_only=False):
        print([(c.coordinate, c.value) for c in row])
"
```

Then run the verifier:
```bash
find / -name 'test_output*' -o -name 'test_*.py' -o -name 'test_*.js' 2>/dev/null
# Run whichever test file exists
```

### 6. Fix any issues

If the verifier fails, read the error message carefully. Common pitfalls from cross-task feedback:
- **Exact header/label text matters** — use the precise strings the verifier expects (the hospital staffing task failed because of a column name mismatch).
- **Cell references must be exact** — if verifier checks B7, the value must be in B7, not B6 or C7.
- **Numeric vs text** — ensure numbers are Python int/float, not strings.
- **Formula syntax** — sheet names with spaces and special chars need single quotes in formulas.
- **Row positioning** — line items start at row 6; control rows follow immediately after the last line item.

Iterate until the verifier passes completely.

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