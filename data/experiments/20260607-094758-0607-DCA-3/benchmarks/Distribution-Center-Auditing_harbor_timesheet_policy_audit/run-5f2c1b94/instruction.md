# Task Instruction

Execute the timesheet compliance audit task end-to-end. Produce two deliverables exactly as specified.

Source: `/root/Timesheet_Submissions.xlsx` containing sheets `Entries` and `BreakRules`.

Deliverables:
1. `/root/Timesheet_Compliance_Audit.xlsx`
2. `/root/Timesheet_Compliance_Brief.docx`

## Step 1: Inspect the source workbook first
- Open `/root/Timesheet_Submissions.xlsx` and print:
  - All columns and dtypes of `Entries`.
  - All rows of `BreakRules` (it provides per-role `Min Break Minutes` and `Overtime Threshold`).
  - A sample of `Entries` including any non-numeric, blank, or sentinel values (e.g. `N/A`, empty strings, NaN) in `Hours Worked`, `Break Minutes`, `Approval Code`.
- Note: prior similar tasks regressed because string literals like `N/A` were coerced to None/NaN when copied. Preserve source cell values literally in `RawData`.

## Step 2: Build `/root/Timesheet_Compliance_Audit.xlsx`

Use openpyxl directly (not pandas to_excel) for `RawData` to guarantee literal preservation, OR read the source with `pandas.read_excel(..., dtype=str, keep_default_na=False, na_values=[])` so that strings like `N/A`, empty strings, and other sentinels are not converted to NaN/None.

### Sheet `RawData`
- Exact copy of `Entries` table (same column order, same headers, same values).
- Do NOT coerce `N/A`, blanks, or any string literal into None/NaN. The cell value written must match the source cell value exactly (string stays string, number stays number).
- Verify after writing: reload `RawData` and assert it equals the source `Entries` cell-by-cell (including string `N/A`).

### Sheet `Formatted Data`
- Same row order as `RawData`.
- First 8 columns identical to `RawData` first 8 columns and headers:
  1. Week Ending, 2. Employee ID, 3. Role, 4. Hours Worked, 5. Break Minutes, 6. Approval Code, 7. Project Code, 8. Manager.
- Preserve the same literal values (do not coerce `N/A`).
- Append columns 9-12 with exact headers: `Break Deficit`, `Approval Missing`, `Total Errors`, `Error Summary`.
- Compute concrete values (no live formulas):
  - Build a role->(min_break, overtime_threshold) lookup from `BreakRules`. Do not hardcode by role name; read whatever roles exist.
  - For each row:
    - Parse `Break Minutes` as numeric; if it is missing/non-numeric, treat as not less than threshold ONLY if the rule clearly intends so — default behavior: if `Break Minutes` cannot be interpreted as a number, treat it as 0 for the comparison (so it counts as a deficit). Apply this consistently; document the choice in code comments. `Break Deficit` = 1 if numeric break < role's `Min Break Minutes`, else 0.
    - Parse `Hours Worked` as numeric similarly (non-numeric -> 0). `Approval Missing` = 1 if hours > role's `Overtime Threshold` AND `Approval Code` is blank (treat empty string, whitespace-only, NaN, or None as blank; `N/A` string literal: treat as a non-blank value unless it is clear from BreakRules/Entries conventions that it means missing — default: treat any non-empty string including `N/A` as NOT blank, i.e. an approval was recorded). If unclear from data, prefer treating only truly empty/NaN cells as blank.
    - `Total Errors` = Break Deficit + Approval Missing (integer).
    - `Error Summary`:
      - 0 errors -> `None`
      - only break deficit -> `Break Deficit`
      - only approval missing -> `Approval Missing`
      - both -> `Break Deficit, Approval Missing`
- Write integers (not floats) for the three numeric added columns.

### Sheet `Summary`
- Headers exactly: `Employee ID`, `Week Ending`, `Break Deficits`, `Approval Gaps`, `Total Errors`.
- Aggregate `Formatted Data` grouped by `(Employee ID, Week Ending)` summing `Break Deficit`, `Approval Missing`, `Total Errors`.
- Keep only groups with `Total Errors > 0`.
- Sort by `Employee ID` ascending, then `Week Ending` ascending.
- Append final row: `Employee ID` = `Grand Total`, `Week Ending` = `-`, remaining three columns are the dataset-wide totals (sum across all rows in `Formatted Data`, not just filtered Summary rows — these should equal the column sums of the included groups since excluded groups have 0).

## Step 3: Build `/root/Timesheet_Compliance_Brief.docx`
Use python-docx. Write 3-6 sentences covering:
- Plain-language definition of `Break Deficit` (break minutes below the role-specific minimum) and `Approval Missing` (overtime hours above the role threshold without an approval code).
- The computed totals: Break Deficits, Approval Gaps, Total Errors (use the Grand Total values).
- At least one actionable recommendation (e.g., require manager pre-approval for overtime, enforce break logging).
- Mention at least two specific high-priority Employee IDs with the most frequent exceptions (top by `Total Errors` from Summary).

## Step 4: Validation before finishing
- Reload the produced xlsx and verify:
  - Sheet names are exactly `RawData`, `Formatted Data`, `Summary` (no extras, no renames).
  - `RawData` matches source `Entries` cell-for-cell including any `N/A` string literals.
  - `Formatted Data` has 12 columns with the exact specified headers.
  - `Error Summary` values are only from the allowed 4 strings.
  - `Total Errors == Break Deficit + Approval Missing` on every row.
  - `Summary` last row is `Grand Total / -` with column sums matching `Formatted Data` column sums.
- Reload the docx and verify it contains the totals and at least two employee IDs.

Report any anomalies found in the source and how you handled them.

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