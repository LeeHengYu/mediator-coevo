# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 – Setup
```bash
mkdir -p /root/output
cp /root/data/workbook.xlsx /root/output/result.xlsx
```

## 1 – Inspect the workbook
Open `/root/output/result.xlsx` with openpyxl and inspect:
- Sheet `Task`: print rows 10-50 (columns D-L) to understand the layout — column D series codes, row 10 year headers, yellow target ranges.
- Sheet `Data`: print rows 21-38 to understand the lookup source — how series codes and years are arranged (which row/column holds what).

Record:
- The exact cell references for the series-code column on `Data` (e.g., column A or B rows 21-38).
- The exact row on `Data` that holds the year headers.
- Confirm the three indicator blocks on `Task`: H12:L17 (Successful API Requests), H19:L24 (Failed API Requests), H26:L31 (Compute Capacity) — verify by reading labels near those rows.
- Confirm rows 35-40 are the six regions for Net reliability gap, rows 42-47 are min/max/median/mean/25th/75th, row 50 is the weighted mean.

## 2 – Write formulas with openpyxl
Use `openpyxl` (do NOT use data_only mode; write formula strings). For every formula, use the `Translator` or manual string building — but the formulas must be Excel-legal strings stored as cell values.

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in those ranges, write an `INDEX/MATCH` formula that:
- Uses the series code from column D of the *same row* on sheet `Task`.
- Uses the year from row 10 of the *same column* on sheet `Task`.
- Looks up in the `Data` sheet rows 21:38.

Concrete pattern (adjust column letters after inspection):
```
=INDEX(Data!$B$21:$B$38, MATCH(1, (Data!$A$21:$A$38=$D12)*(Data!$B$20:$XX$20=H$10), 0))
```
However, since INDEX/MATCH for a 2-D lookup typically needs both a row match and a column match, use this canonical form:
```
=INDEX(Data!<data_area>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```
Replace `<data_area>`, `<series_code_column>`, and `<year_header_row>` with the actual references found during inspection. Use absolute references ($) for the Data ranges and mixed references ($D12 for row-relative, H$10 for column-relative) so the formula copies correctly across the 5×6 block.

Write the formula into every cell of each block. Do NOT hardcode values.

### Step 2 – Net reliability gap (H35:L40)
For each of the 6 region rows and 5 year columns:
```
=(H12-H19)/H26*100
```
Adjust row references per region (row 35 uses rows 12,19,26; row 36 uses 13,20,27; etc.).

Statistics in H42:L47 (column-wise over H35:L40):
- H42: `=MIN(H35:H40)`
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)`
- H46: `=PERCENTILE(H35:H40,0.25)`  ← use legacy PERCENTILE, NOT PERCENTILE.INC
- H47: `=PERCENTILE(H35:H40,0.75)`

Copy across columns H-L.

### Step 3 – Weighted mean (H50:L50)
For each column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## 3 – Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets.

## 4 – Verify
Reopen the saved file and print the formula strings in a few sample cells (e.g., H12, L17, H35, H42, H46, H50) to confirm they are stored as expected formula strings (starting with `=`). Confirm no cells contain raw Python values where formulas are expected.

## Important Notes
- Use `PERCENTILE` (legacy), not `PERCENTILE.INC` or `PERCENTILE.EXC`.
- Do not use array-entry (CSE) formulas; standard INDEX/MATCH is sufficient.
- Preserve all existing formatting — do not touch fonts, fills, borders, column widths, or any cells outside the target ranges.
- Do not add macros, VBA, external links, helper columns, or extra sheets.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=easy, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.