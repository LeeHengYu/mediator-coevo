# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 – Environment Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1 – Inspect the workbook
Open `/root/data/workbook.xlsx` with `openpyxl` (data_only=False) and print:
- On sheet `Data`: the contents of column A/B rows 20–40 (to find series codes), and row 20 or 21 columns A–N (to find year headers). Print exact cell references and values.
- On sheet `Task`: the contents of column D rows 12–31 (series codes used for lookup), row 10 columns H–L (year headers), and the labels in column D rows 35–40 and 42–47, and D50.

Print everything so we know the exact layout before writing formulas.

## 2 – Write lookup formulas in H12:L31
Using `openpyxl`, write `INDEX/MATCH` formulas into every cell in the three blocks `H12:L17`, `H19:L24`, `H26:L31`.

For each cell at row `r`, column `c` (H=8 … L=12):
- The series code is in `$D{r}` on sheet `Task`.
- The year is in `{col_letter}$10` on sheet `Task` (where col_letter corresponds to column c).
- The data table is on sheet `Data`, rows 21:38. Identify which column holds the series codes and which row holds the year headers from the inspection above.

Suppose the Data sheet has series codes in column B (rows 21–38) and year values in row 20 (or 21) starting from some column. Construct the formula pattern:
```
=INDEX(Data!C21:C38,MATCH($D{r},Data!$B$21:$B$38,0))
```
But we need a 2D lookup: match the series code to find the row, and match the year to find the column. So use:
```
=INDEX(Data!$C$21:$XX$38, MATCH($D{r},Data!$B$21:$B$38,0), MATCH({col_letter}$10,Data!$C$20:$XX$20,0))
```
Adjust the exact column letters and row numbers based on what the inspection reveals. The key contract: the INDEX range must cover the data values (not include the header column/row), the first MATCH finds the row by series code, the second MATCH finds the column by year.

## 3 – Write Net patient flow formulas in H35:L40
For each hospital row `r_out` in 35–40, the corresponding rows are:
- Patient Admissions: rows 12–17 (block 1)
- Patient Discharges: rows 19–24 (block 2)
- Effective Bed Capacity: rows 26–31 (block 3)

So hospital index `i` (0–5) maps to:
- Admissions row = 12 + i
- Discharges row = 19 + i
- Capacity row = 26 + i
- Output row = 35 + i

For each cell at (r_out, col_letter):
```
=({col_letter}{adm_row}-{col_letter}{dis_row})/{col_letter}{cap_row}*100
```
Use parentheses exactly as shown.

## 4 – Write summary statistics in H42:L47
For each column col_letter in H–L:
- Row 42 (Min): `=MIN({col_letter}35:{col_letter}40)`
- Row 43 (Max): `=MAX({col_letter}35:{col_letter}40)`
- Row 44 (Median): `=MEDIAN({col_letter}35:{col_letter}40)`
- Row 45 (Mean): `=AVERAGE({col_letter}35:{col_letter}40)`
- Row 46 (25th pct): `=PERCENTILE({col_letter}35:{col_letter}40,0.25)`
- Row 47 (75th pct): `=PERCENTILE({col_letter}35:{col_letter}40,0.75)`

**Important**: Use `PERCENTILE` (not `PERCENTILE.INC`) to avoid `#NAME?` errors in openpyxl. If the verifier expects `PERCENTILE.INC`, prefix with `_xlfn.` like `=_xlfn.PERCENTILE.INC(...)`. Try `PERCENTILE` first as it is more compatible.

## 5 – Write weighted mean in H50:L50
For each column col_letter in H–L:
```
=SUMPRODUCT({col_letter}35:{col_letter}40,{col_letter}26:{col_letter}31)/SUM({col_letter}26:{col_letter}31)
```

## 6 – Save
Save the workbook to `/root/output/result.xlsx`. Do NOT call `data_only=True` when loading. Do NOT add sheets, macros, or VBA.

## 7 – Verify
Reload the saved file (data_only=False) and print the values of cells H12, H19, H26, H35, H42, H50 to confirm they contain formula strings (starting with `=`), not None.

## Critical Reminders
- Assign formulas to `cell.value`, e.g., `ws['H12'] = '=INDEX(...)'`. Do NOT use `.formula` attribute.
- Make sure you reference the correct sheet name in formulas. If the Data sheet tab is named exactly `Data`, use `Data!` prefix. Check the actual sheet name from `wb.sheetnames`.
- Do not modify any existing formatting, sheets, or structure.
- Double-check every formula references the correct rows/columns from the inspection step before writing.

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