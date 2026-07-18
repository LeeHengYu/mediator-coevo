# Task Instruction

## Task: Build Aurora Rights Rollforward Excel Workbook

You must create an Excel workbook at `/root/Aurora_Rights_Rollforward_4-25.xlsx` with exactly three sheets in order: `Rights Summary`, `Film Rights #2710`, `Music Rights #2720`.

### Step 0: Inspect all input files

1. Read and print `/root/film_rights_schedule_input.csv` completely.
2. Read and print `/root/music_rights_schedule_input.csv` completely.
3. Read and print `/root/rights_ledger_balances.json` completely.
4. Read the three context/source documents to understand any operational notes:
   - `/root/Aurora_Film_Licensor_Invoices_Q1Q2_2025.txt`
   - `/root/Aurora_Music_Licensor_Invoices_Q1Q2_2025.txt`
   - `/root/aurora_rights_ledger_control_notes_apr25.txt`
5. List all files in `/root/` to check for any verifier scripts, test files, or reference files.
6. If there is a `tests/` directory or any `.py` test files, read them completely to understand the verifier contract.

### Step 1: Understand the structure

Based on the task description, the workbook follows a "Harbor reconciliation" pattern:

**Detail sheets (`Film Rights #2710` and `Music Rights #2720`):**
- Row 1-5: Headers (column headers likely in row 5, with months as columns)
- Row 6+: Line items from the CSV input files (one row per licensor/rights item)
- After line items, control rows in this order:
  - `Month Totals` — SUM of the line-item rows above, per month column
  - `Ending Balance` — running cumulative balance
  - `Variance` — difference between Ending Balance and GL Balance
  - `GL Balance` — from the ledger balances JSON
- Column A: labels/names
- Columns B onward (likely B through O or similar): months
- Column O appears to be a key summary column (possibly a total or final month)

**Summary sheet (`Rights Summary`):**
- B7, B8, B9: Film rights summary values linked to `Film Rights #2710` column O
- B12, B13, B14: Music rights summary values linked to `Music Rights #2720` column O
- B16: Combined total = B9 + B14
- The exact meaning of rows 7/8/9 and 12/13/14 likely maps to control-row values (e.g., Ending Balance, GL Balance, Variance or similar groupings)

### Step 2: Determine exact layout from the data

After inspecting the CSV files:
- Identify column headers (months, totals)
- Identify how many line items exist
- Identify the month columns and which column index corresponds to column O (index 15, i.e., the 14th data column after column A)
- Map the JSON ledger balances to the GL Balance row

### Step 3: Build the workbook using openpyxl

Use Python with openpyxl:

```python
import openpyxl
import csv
import json
```

For each detail sheet:
1. Write header rows (rows 1-5) with appropriate titles and month column headers.
2. Starting at row 6, write each line item from the corresponding CSV. Ensure all numeric values are stored as numbers (float/int), NOT as strings.
3. After the last line item row, add control rows:
   - **Month Totals**: For each month column, use an Excel SUM formula referencing row 6 through the last line-item row. E.g., `=SUM(B6:B{last_item_row})`
   - **Ending Balance**: Compute as Beginning Balance + Month Totals (or as specified by the data pattern — inspect the CSV to determine if there's a beginning balance row)
   - **Variance**: `=Ending Balance row - GL Balance row` for each column
   - **GL Balance**: Insert values from the JSON ledger data
4. Column A of each control row must contain exactly: `Month Totals`, `Ending Balance`, `Variance`, `GL Balance`.

For the Rights Summary sheet:
- B7: `='Film Rights #2710'!O{ending_balance_row}` (or whichever control row maps here)
- B8: `='Film Rights #2710'!O{gl_balance_row}`
- B9: `='Film Rights #2710'!O{variance_row}`
- B12: `='Music Rights #2720'!O{ending_balance_row}`
- B13: `='Music Rights #2720'!O{gl_balance_row}`
- B14: `='Music Rights #2720'!O{variance_row}`
- B16: `=B9+B14`

**Important**: The exact mapping of B7/B8/B9 and B12/B13/B14 to which control rows depends on what the data and verifier expect. Inspect test files carefully to determine the correct mapping. If no test files clarify this, use the most logical mapping: B7=Ending Balance, B8=GL Balance, B9=Variance (and similarly for B12-B14).

### Step 4: Validate

1. Re-open the workbook with openpyxl (data_only=False) and verify:
   - Exactly 3 sheets in the correct order
   - Sheet names are exactly `Rights Summary`, `Film Rights #2710`, `Music Rights #2720`
   - Line items start at row 6 in detail sheets
   - Control row labels are present and in the correct rows
   - Formulas exist in Month Totals, Variance rows
   - Summary sheet B7/B8/B9/B12/B13/B14/B16 contain formulas referencing detail sheets
   - All numeric cells are numeric type, not string
2. If any verifier/test script exists, run it and fix any issues.
3. Confirm the file exists at `/root/Aurora_Rights_Rollforward_4-25.xlsx`.

### Critical Rules
- Do NOT modify any source/input files.
- All numeric values must be stored as numbers (int or float), never as strings.
- Use Excel formulas (not hardcoded values) for Month Totals sums, Ending Balance calculations, Variance calculations, and all Summary sheet references.
- The summary formulas in B7/B8/B9/B12/B13/B14 must reference column O of the respective detail tabs.
- B16 must be the formula `=B9+B14`.
- Sheet order matters: Rights Summary first, then Film, then Music.

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

Task-local resources are available under `environment/skills`: invoice-organizer, monthly-close.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=noreply@example.com, author_name=Codex Task Generator, category=media-operations, difficulty=medium, tags=[excel, media-rights, invoice-normalization, reconciliation, rollforward].
Verifier config: timeout_sec=900.0.