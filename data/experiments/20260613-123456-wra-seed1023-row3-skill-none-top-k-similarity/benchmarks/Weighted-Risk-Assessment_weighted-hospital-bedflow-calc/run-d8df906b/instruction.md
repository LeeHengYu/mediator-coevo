# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
cp /root/data/workbook.xlsx /root/output/result.xlsx
```

## 1. Inspect the workbook structure
Open `/root/output/result.xlsx` with openpyxl and inspect:
- Sheet names
- Sheet `Task`: print rows 1-55, columns A-L, to understand layout (especially column D series codes, row 10 years, yellow cell regions)
- Sheet `Data`: print rows 1-40 to understand the data layout (especially rows 21-38)

Print cell values so you can see:
- What series codes are in column D for rows 12-17, 19-24, 26-31
- What years are in H10:L10
- The structure of Data sheet rows 21-38 (what's in each column, how series codes and years are arranged)

## 2. Populate H12:L17, H19:L24, H26:L31 with lookup formulas

For each cell in these ranges, write a formula that looks up data from `Data!$21:$38` using:
- The series code from column D of the current row on sheet `Task`
- The year from row 10 of the current column on sheet `Task`

Based on the Data sheet layout you discovered in step 1, choose the appropriate lookup pattern. The most likely patterns:

**If Data rows 21-38 have series codes in one column and years across columns (horizontal layout):**
Use `INDEX(MATCH, MATCH)` pattern:
`=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))`

**If Data rows 21-38 have data in a vertical/tabular layout with a series code column and a year column:**
Use a combination like `INDEX(MATCH(1, (code_col=$D12)*(year_col=H$10), 0))` or nested lookups.

IMPORTANT: Lock references appropriately:
- Column D reference: use `$D12` (lock column, relative row)
- Row 10 reference: use `H$10` (relative column, lock row)
- Data range references: use absolute references with sheet name `Data!`

Write the formulas using openpyxl. When writing formulas as strings in openpyxl, do NOT include a leading `=` in the string — actually, openpyxl DOES require the leading `=`. Write the formula as a string starting with `=`.

Make sure to preserve existing formatting. Use `cell.value = '=FORMULA...'` only — do not touch `cell.font`, `cell.fill`, `cell.border`, `cell.number_format`, `cell.alignment`, etc.

## 3. Populate H35:L40 with Net Patient Flow formulas

Net patient flow = (Patient Admissions - Patient Discharges) / Effective Bed Capacity * 100

Based on the layout:
- H12:L17 likely contains one metric (e.g., Patient Admissions)
- H19:L24 likely contains another metric (e.g., Patient Discharges)
- H26:L31 likely contains Effective Bed Capacity

Verify which block is which by reading column D labels or nearby labels. Then for each cell in H35:L40:
`=(H12-H19)/H26*100` (adjusting row references for each hospital row)

Make sure the hospital order in rows 35-40 matches the order in rows 12-17, 19-24, 26-31.

## 4. Populate H42:L47 with summary statistics

For each column H through L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

Verify the order (min/max/median/mean/25th/75th) by checking any labels in column D or nearby columns for rows 42-47.

## 5. Populate H50:L50 with weighted mean using SUMPRODUCT

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of Net Patient Flow using Effective Bed Capacity as weights.

## 6. Save
Save with openpyxl. Use `wb.save('/root/output/result.xlsx')`. Make sure to open with `data_only=False` (default) so formulas are preserved.

## 7. Verify
Reopen the file and print the formula content of a sample of cells (e.g., H12, L17, H35, H40, H42, H47, H50, L50) to confirm formulas were written correctly.

## CRITICAL NOTES
- Do NOT use `load_workbook(..., data_only=True)` — that strips formulas.
- Do NOT modify cell formatting (font, fill, border, alignment, number_format).
- Do NOT add new sheets.
- Do NOT delete any existing content outside the specified cell ranges.
- Carefully inspect the Data sheet layout before writing formulas — the exact column/row references in your formulas depend on how the data is arranged.
- If the Data sheet has a matrix layout with series codes in one column and years in a header row, INDEX/MATCH with two MATCH functions is ideal.
- Use `PERCENTILE.INC` or `PERCENTILE` — try `PERCENTILE` first as it's more universally supported in xlsx.

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