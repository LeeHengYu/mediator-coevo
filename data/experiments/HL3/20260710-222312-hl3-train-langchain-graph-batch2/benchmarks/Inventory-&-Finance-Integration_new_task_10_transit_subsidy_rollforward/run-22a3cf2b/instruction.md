# Task Instruction

Build the Excel workbook `/root/MetroLink_Pass_Liability_4-25.xlsx` using Python (openpyxl). Follow these steps precisely:

## Step 1: Inspect all input files

```bash
cat /root/bus_pass_schedule_input.csv
cat /root/rail_pass_schedule_input.csv
cat /root/fare_liability_balances.json
cat /root/MetroLink_Bus_Pass_Issuance_Notes_Q1Q2_2025.txt
cat /root/MetroLink_Rail_Pass_Issuance_Notes_Q1Q2_2025.txt
cat /root/metrolink_fare_ledger_control_notes_apr25.txt
```

Read every file completely before writing any code. Understand:
- What months/columns are in the CSV schedules
- What the JSON balances contain (opening balances, GL balances, etc.)
- What the text notes say about the operational context

## Step 2: Understand the required structure

The workbook must have exactly 3 sheets in this order:
1. `Transit Summary`
2. `Bus Program #4310`
3. `Rail Program #4320`

### Detail sheets (`Bus Program #4310` and `Rail Program #4320`) structure:
- **Row 1-5**: Headers (title, column headers, etc.)
- **Row 6 onward**: Line items start at row 6. These are the monthly data rows from the CSV inputs.
- **Columns**: Column A = row labels, Columns B through N (or similar) = monthly data, Column O = totals/summary column (likely a SUM across the monthly columns for each row)
- **Control rows** (after the line items):
  - `Month Totals` row: sums each column across the line item rows
  - `Ending Balance` row: computed from opening balance + month totals (or a rollforward formula)
  - `Variance` row: difference between ending balance and GL balance
  - `GL Balance` row: from the JSON fare_liability_balances

### `Transit Summary` sheet structure:
- This sheet summarizes both programs
- Key formula cells:
  - **B7** = links to column O of `Bus Program #4310` (likely the ending balance or a key total)
  - **B8** = links to column O of `Bus Program #4310` (another key metric)
  - **B9** = links to column O of `Bus Program #4310` (another key metric)
  - **B12** = links to column O of `Rail Program #4320`
  - **B13** = links to column O of `Rail Program #4320`
  - **B14** = links to column O of `Rail Program #4320`
  - **B16** = formula `=B9+B14` (combined total from both programs)

## Step 3: Map CSV data to the detail sheets

For each CSV file:
- Parse the CSV to understand its columns and rows
- The CSV likely has months as columns and pass types/categories as rows
- Map these into the detail sheet starting at row 6
- Column A = category/label, subsequent columns = monthly values, Column O = row totals (use SUM formulas)
- After the line items, add the control rows: `Month Totals`, `Ending Balance`, `Variance`, `GL Balance`

## Step 4: Map JSON data

The `fare_liability_balances.json` likely contains:
- Opening balances for each program
- GL balances for reconciliation
- Use these for the `GL Balance` control row and for computing `Ending Balance`

## Step 5: Build the workbook with Python

Use openpyxl. Key requirements:
- All numeric values must be stored as numbers (int or float), NOT as strings
- Use Excel formulas (strings starting with `=`) for computed cells, especially:
  - Column O totals on detail sheets (e.g., `=SUM(B6:N6)`)
  - Month Totals rows (e.g., `=SUM(B6:B15)` or similar vertical sums)
  - Ending Balance = Opening Balance + Month Totals (or appropriate rollforward)
  - Variance = Ending Balance - GL Balance
  - Summary sheet B7/B8/B9 referencing `'Bus Program #4310'!O<row>`
  - Summary sheet B12/B13/B14 referencing `'Rail Program #4320'!O<row>`
  - B16 = `=B9+B14`
- Do NOT modify any source files

## Step 6: Determine exact row mappings

After reading the CSV files, determine:
- How many line item rows there are (starting at row 6)
- Which row numbers the control rows land on
- Which specific rows in the detail sheets correspond to B7/B8/B9 and B12/B13/B14 in the summary

The summary cell references should be:
- B7 → `'Bus Program #4310'!O<row>` for one of the control rows (likely Month Totals, Ending Balance, or a specific metric)
- B8 → another control row from Bus
- B9 → another control row from Bus
- B12 → corresponding row from Rail
- B13 → corresponding row from Rail  
- B14 → corresponding row from Rail
- B16 → `=B9+B14`

Look at the data to determine the logical mapping. The pattern B7/B8/B9 for Bus and B12/B13/B14 for Rail with B16=B9+B14 suggests:
- Rows 7-9 are Bus program summary (3 key figures)
- Rows 12-14 are Rail program summary (3 key figures)
- Row 16 is combined total
- Rows 6, 10-11, 15 might be labels/headers/spacers

## Step 7: Validate

After creating the workbook:
1. Reopen it with openpyxl and verify:
   - Exactly 3 sheets in the correct order
   - Sheet names match exactly: `Transit Summary`, `Bus Program #4310`, `Rail Program #4320`
   - Line items start at row 6 on detail sheets
   - Control rows exist with correct labels
   - B7/B8/B9/B12/B13/B14 on Transit Summary contain formulas referencing column O of detail sheets
   - B16 contains formula `=B9+B14`
   - All numeric cells contain numbers, not strings
2. Print the cell values and types for key cells to confirm

## Critical Notes
- Read ALL input files first before writing any code
- The exact row numbers for control rows depend on how many line items come from the CSVs
- Adapt the structure based on what you find in the data, but maintain the required pattern
- If the CSV has an 'Opening Balance' or 'Beginning Balance' row, that likely goes in row 5 (before line items at row 6)
- Column O must be the summary/total column on detail sheets
- Do not guess at data values - use exactly what's in the input files

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