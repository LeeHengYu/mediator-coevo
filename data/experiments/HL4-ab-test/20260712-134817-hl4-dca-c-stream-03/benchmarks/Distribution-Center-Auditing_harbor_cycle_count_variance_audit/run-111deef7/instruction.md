# Task Instruction

Execute the following steps in a single Python script to produce the two deliverables.

## 0 – Inspect inputs
```
import openpyxl, os
for f in ['Cycle_Plan.xlsx','Count_Event_Log.xlsx','Cycle_Template.xlsx']:
    wb = openpyxl.load_workbook(f'/root/{f}')
    print(f, wb.sheetnames)
    for s in wb.sheetnames:
        ws = wb[s]
        print(f'  {s}: {ws.max_row} rows x {ws.max_column} cols')
        for r in ws.iter_rows(min_row=1, max_row=min(5, ws.max_row), values_only=True):
            print('   ', r)
    wb.close()
```
Run this first, read the output carefully, then proceed.

## 1 – Build `/root/Cycle_Count_Variance_Audit.xlsx`

Use `openpyxl` for the Overview copy and `pandas` + `openpyxl` for the data sheets.

### 1a) Overview sheet
- Load `Cycle_Template.xlsx` with openpyxl (data_only=False to preserve formulas if any).
- Copy the `Overview` sheet cell-by-cell (values, styles if feasible, merged cells) into the output workbook's first sheet named `Overview`.
- Preserve it unchanged.

### 1b) RawData sheet
- Read `Cycle_Plan.xlsx` into a pandas DataFrame (use the header row you observed in step 0).
- Write it verbatim to the `RawData` sheet, preserving column names and row order.

### 1c) Formatted Data sheet
- Start from the RawData DataFrame. Rename columns if needed so the first 7 columns are exactly:
  1. Facility
  2. Session ID
  3. Bin ID
  4. Product ID
  5. Expected Qty
  6. Allowed Variance
  7. Approval Needed
- Process `Count_Event_Log.xlsx`:
  - Read it into a DataFrame. Print its columns and first rows.
  - Keep only rows where `Event Type` (or equivalent column) equals `FINAL` (case-insensitive after stripping).
  - Drop rows where any of Facility, Session ID, Bin ID, or Count Qty is blank/NaN.
  - Sort by timestamp or row order descending, then drop duplicates on `(Facility, Session ID, Bin ID)` keeping last (i.e., the latest FINAL row per key). If there is a timestamp column use it; otherwise use original row order (index) as proxy.
  - Build a lookup dict keyed by `(Facility, Session ID, Bin ID)` → `Count Qty`.
- For each row in the plan DataFrame, compute:
  - `Missing Final Count`: 1 if key not in lookup, else 0.
  - `Approval Gap`: 1 if ALL of:
    - key IS in lookup (final count exists),
    - `Approval Needed` stripped upper == 'YES',
    - abs(Expected Qty − Count Qty) > Allowed Variance
    Otherwise 0.
  - `Total Errors` = Missing Final Count + Approval Gap.
  - `Error Summary`: build from the two flags:
    - both 0 → 'None'
    - only Missing → 'Missing Final Count'
    - only Approval → 'Approval Gap'
    - both → 'Missing Final Count, Approval Gap'
- Write to `Formatted Data` sheet with columns 1-11 as specified. Write concrete values (int/str), not formulas.

### 1d) Summary sheet
- From `Formatted Data` DataFrame, group by `(Facility, Session ID)`.
- Aggregate: Missing Final Counts = sum, Approval Gaps = sum, Total Errors = sum.
- Filter to groups with Total Errors > 0.
- Sort by Facility ascending then Session ID ascending.
- Append a Grand Total row: Facility='Grand Total', Session ID='-', sums of the three numeric columns across the kept groups (i.e., dataset totals from the full Formatted Data, not just filtered groups — re-check: the instruction says "dataset totals", so sum from ALL rows of Formatted Data, not just the filtered summary rows).
- Write with headers: Facility, Session ID, Missing Final Counts, Approval Gaps, Total Errors.

### 1e) Save
- Use openpyxl to write all sheets into one workbook. Strategy:
  - Create the workbook with openpyxl, copy Overview first.
  - Then use `pandas.ExcelWriter` with `openpyxl` engine and `mode='a', if_sheet_exists='replace'` to add the three data sheets.
  - Or build everything in openpyxl directly.
- Save as `/root/Cycle_Count_Variance_Audit.xlsx`.
- Verify: reopen and print sheet names, row counts, and first few rows of each sheet.

## 2 – Build `/root/Cycle_Count_Variance_Brief.docx`

Use `python-docx`. Install if needed: `pip install python-docx`.

Write 3-6 sentences that include:
- Plain-language definition of Missing Final Count check (a bin in the cycle plan has no confirmed final physical count recorded in the event log).
- Plain-language definition of Approval Gap check (a bin's final counted quantity deviates from the expected quantity beyond the allowed variance threshold, yet requires managerial approval).
- The exact computed totals for Missing Final Counts, Approval Gaps, and Total Errors (use the Grand Total values).
- At least one actionable recommendation (e.g., prioritize recounts, investigate root causes, tighten count procedures).
- Mention at least two specific high-priority (Facility, Session ID) combinations that have the most exceptions (pick the top 2 by Total Errors from the Summary table).

Save as `/root/Cycle_Count_Variance_Brief.docx`.

## 3 – Final Verification
- Confirm both files exist.
- Reopen the Excel file and print all sheet names, column headers, row counts, and a sample of rows from each sheet.
- Reopen the Word file and print all paragraph text to confirm content.
- If anything looks wrong, fix it before finishing.

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

Inspect the task files, environment, tests, and expected outputs directly.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=expert, tags=[excel, openpyxl, docx, audit, inventory].
Verifier config: timeout_sec=900.0.