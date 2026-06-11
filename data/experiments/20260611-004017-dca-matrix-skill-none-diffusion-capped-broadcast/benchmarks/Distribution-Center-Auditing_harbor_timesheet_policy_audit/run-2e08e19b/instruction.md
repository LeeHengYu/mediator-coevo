# Task Instruction

## Task: Timesheet Compliance Audit

### Goal
Read `/root/Timesheet_Submissions.xlsx`, perform compliance checks, and produce two deliverables:
1. `/root/Timesheet_Compliance_Audit.xlsx`
2. `/root/Timesheet_Compliance_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect the source workbook
- Read `/root/Timesheet_Submissions.xlsx` and list the sheet names.
- Print the first 5 rows and all column headers of the `Entries` sheet.
- Print the entire `BreakRules` sheet (it should be small — role-based thresholds).
- Note the exact column names in both sheets. Do NOT assume column names; use what you see.

#### Step 1: Install dependencies if needed
```bash
pip install openpyxl python-docx pandas
```

#### Step 2: Write and run a Python script that does everything below

Use pandas + openpyxl for Excel and python-docx for Word.

##### 2a: Load data
- Load `Entries` sheet into a DataFrame `df_raw`.
- Load `BreakRules` sheet into a DataFrame `df_rules`.
- Identify the column in `BreakRules` that contains the role name, the minimum break minutes, and the overtime threshold. Use the actual column names you observed in Step 0.

##### 2b: Build `RawData` sheet
- This is an exact copy of the `Entries` table. Write `df_raw` to sheet `RawData` (index=False).

##### 2c: Build `Formatted Data` sheet
- Start from a copy of `df_raw`.
- Merge/map the `BreakRules` thresholds onto each row by `Role`.
- Compute four new columns with EXACTLY these header strings:
  - `Break Deficit`: 1 if `Break Minutes` < `Min Break Minutes` for that role, else 0
  - `Approval Missing`: 1 if `Hours Worked` > `Overtime Threshold` for that role AND `Approval Code` is blank/NaN, else 0
  - `Total Errors`: `Break Deficit` + `Approval Missing`
  - `Error Summary`: exactly one of these four strings:
    - `None` (if Total Errors == 0)
    - `Break Deficit` (if only break deficit)
    - `Approval Missing` (if only approval missing)
    - `Break Deficit, Approval Missing` (if both)
- The first 8 columns must be exactly: Week Ending, Employee ID, Role, Hours Worked, Break Minutes, Approval Code, Project Code, Manager — in that order. If the source columns are in a different order, reorder them. If the source column names differ slightly, rename them to match these exact names.
- Columns 9-12 are the four new columns above.
- Keep the same row order as RawData.
- Write concrete values (int/string), not formulas.
- Write to sheet `Formatted Data` (index=False).

##### 2d: Build `Summary` sheet
- From the `Formatted Data` DataFrame, group by (`Employee ID`, `Week Ending`).
- Aggregate: sum of `Break Deficit` → `Break Deficits`, sum of `Approval Missing` → `Approval Gaps`, sum of `Total Errors` → `Total Errors`.
- Filter to only groups where `Total Errors > 0`.
- Sort by `Employee ID` ascending, then `Week Ending` ascending.
- Append a final row: `Employee ID` = `Grand Total`, `Week Ending` = `-`, and the remaining three columns = the column totals across all included rows.
- Column headers must be exactly: `Employee ID`, `Week Ending`, `Break Deficits`, `Approval Gaps`, `Total Errors`.
- Write to sheet `Summary` (index=False).

##### 2e: Save the Excel workbook
- Save to `/root/Timesheet_Compliance_Audit.xlsx` using `openpyxl` engine.
- The workbook must contain exactly three sheets: `RawData`, `Formatted Data`, `Summary`.

##### 2f: Build the Word document
- Create `/root/Timesheet_Compliance_Brief.docx` with a short executive summary (3-6 sentences).
- Include:
  - A plain-language definition of both checks (Break Deficit: employee took less break than the minimum required for their role; Approval Missing: employee worked overtime beyond their role's threshold without an approval code).
  - The computed Grand Total numbers for Break Deficits, Approval Gaps, and Total Errors (use the actual numbers from the Summary Grand Total row).
  - At least one actionable recommendation (e.g., enforce mandatory break logging, require pre-approval for overtime).
  - Mention at least two specific Employee IDs that appear most frequently in the Summary (i.e., have the highest Total Errors). Pick the top 2 by Total Errors (excluding Grand Total row).

#### Step 3: Validate outputs
- Re-open `/root/Timesheet_Compliance_Audit.xlsx` and confirm:
  - Sheet names are exactly `['RawData', 'Formatted Data', 'Summary']`.
  - `RawData` row count matches source `Entries`.
  - `Formatted Data` has 12 columns with correct headers.
  - `Summary` last row has `Employee ID` == `Grand Total`.
  - Print the Summary sheet contents for visual check.
- Re-open `/root/Timesheet_Compliance_Brief.docx` and print its text to confirm it contains the required elements.

### Critical Constraints
- Use thresholds from `BreakRules` dynamically (do not hardcode break minutes or overtime thresholds by role name).
- Output filenames and sheet names must be EXACTLY as specified — case-sensitive, spaces included.
- `Approval Code` is considered blank if it is NaN, None, or empty string.
- All numeric flag columns (Break Deficit, Approval Missing, Total Errors) must be integers (0 or 1 per row, sums in Summary).
- `Error Summary` strings must match exactly (watch comma+space formatting).
- In the Summary, ensure `Break Deficits` and `Approval Gaps` are the column names (plural), not `Break Deficit` / `Approval Missing`.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=hard, tags=[excel, openpyxl, docx, audit, timesheet].
Verifier config: timeout_sec=900.0.