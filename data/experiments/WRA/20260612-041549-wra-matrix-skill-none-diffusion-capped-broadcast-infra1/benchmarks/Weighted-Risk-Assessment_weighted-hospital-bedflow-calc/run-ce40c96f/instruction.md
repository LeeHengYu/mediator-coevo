# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl and inspect:
- Sheet names
- Sheet `Task`: Print cells A10:L50 to understand the layout — column headers in row 10, series codes in column D, hospital names, yellow cell regions, etc.
- Sheet `Data`: Print rows 21:38 to understand the source data layout (column headers, row labels, structure).
- Note the exact years in `Task!H10:L10` and the exact series codes in `Task!D12:D17`, `D19:D24`, `D26:D31`.
- Note the structure of `Data!A21:Z38` (or however wide it extends) — which row has headers, which column has series codes, which columns have year data.

Print all of this before writing any formulas.

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each yellow cell at row `r`, column `c` (H=8, I=9, J=10, K=11, L=12):
- The series code is in `Task!D{r}`
- The year is in `Task!{col_letter}10` (e.g., H10, I10, etc.)
- The data source is `Data!A21:XX38` (determine exact range from inspection)

Use INDEX/MATCH formulas. The exact formula pattern depends on the Data sheet layout discovered in step 1. Two likely patterns:

**If Data has series codes in a column and years in a header row:**
```
=INDEX(Data!$B$22:$Z$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$Z$21, 0))
```
Adjust ranges based on actual inspection.

**If Data uses VLOOKUP-friendly layout (codes in first column, years across top):**
```
=INDEX(Data_range, MATCH(series_code_ref, code_column, 0), MATCH(year_ref, header_row, 0))
```

Write these formulas using openpyxl by setting each cell's value to the formula string (prefixed with `=`). Make sure:
- Row references for the series code use `$D{r}` (absolute column, relative row)
- Column references for the year use `{col}$10` (relative column, absolute row)
- Data range references are fully absolute with sheet prefix `Data!`
- Use exact 0 for match_type (exact match)

Apply formulas to all 3 blocks (6 rows × 5 columns each = 30 cells per block, 90 cells total).

## 3. Net Patient Flow in H35:L40

Based on the layout, rows 35-40 correspond to the six hospitals. The formula for each cell is:
```
=(H12 - H19) / H26 * 100
```
where H12 is Patient Admissions, H19 is Patient Discharges, H26 is Effective Bed Capacity for the same hospital and year. Adjust row offsets: if hospital 1 is row 12/19/26/35, hospital 2 is row 13/20/27/36, etc.

So for cell at row `r` in 35-40, column `c`:
```
=({col}{r-23} - {col}{r-16}) / {col}{r-9} * 100
```
Verify the offset arithmetic against the actual row positions found in step 1.

## 4. Summary statistics in H42:L47

For each column (H through L), compute column-wise stats over H35:L40:
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)`

Check the actual labels in column A/B/C rows 42-47 to confirm the correct order of statistics. Adjust row assignments if the order differs.

## 5. Weighted mean in H50:L50

For each column `c` (H through L):
```
=SUMPRODUCT({c}35:{c}40, {c}26:{c}31) / SUM({c}26:{c}31)
```
This computes the weighted mean of Net Patient Flow using Effective Bed Capacity as weights.

## 6. Save

Save the workbook to `/root/output/result.xlsx` using openpyxl. Do NOT change any existing formatting, do NOT add sheets, macros, or VBA.

## 7. Verify

Reopen `/root/output/result.xlsx` and print:
- A sample of the formula cells (e.g., H12, L17, H35, L40, H42, H47, H50, L50) to confirm formulas are correctly written as strings.
- Confirm no extra sheets were added.
- Confirm the formulas reference the correct cells.

## CRITICAL NOTES
- You MUST inspect the workbook thoroughly before writing any formulas. The exact cell references depend on the actual layout.
- All formulas must be Excel formula strings (starting with `=`), not computed Python values.
- Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) for maximum compatibility.
- Do not modify any existing cell content or formatting outside the specified yellow cell ranges.
- When loading with openpyxl, do NOT use `data_only=True` — load with formulas preserved.
- Make sure to preserve any existing content in non-target cells by only writing to the specified ranges.

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