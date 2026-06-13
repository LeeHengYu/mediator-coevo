# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 – Environment & Inspection
```bash
mkdir -p /root/output
pip install openpyxl
```
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and inspect:
- Sheet `Task`: read cell D12:D17 (series codes for block 1), D19:D24 (block 2), D26:D31 (block 3). Read row 10 columns H–L (years). Note exact text/values.
- Sheet `Data`: read rows 21–38 to understand the layout (which column holds the series code, which row holds years, orientation of the data).
Print all of this so you know the exact structure before writing any formulas.

## 1 – Lookup Formulas (H12:L17, H19:L24, H26:L31)

Use `INDEX/MATCH` (two-dimensional lookup). For each yellow cell at row `r`, column `c` (H=8 … L=12):

```
=INDEX(Data!<data_range>, MATCH(<Task cell with series code>, Data!<series_code_column>, 0), MATCH(<Task cell with year>, Data!<year_row>, 0))
```

Concretely, after inspecting the Data sheet:
- Identify the rectangular data range on `Data` rows 21–38 (excluding headers).
- Identify the column that holds series codes (likely column A or B on Data) and the row that holds years (likely row 21 or the row just above the data).
- Build absolute references for the data range, series-code column, and year row.
- The series code reference should lock the column (e.g., `$D12`) and the year reference should lock the row (e.g., `H$10`) so formulas can be filled across the 5×6 blocks.

Write these formulas into all three blocks (H12:L17, H19:L24, H26:L31) using openpyxl. Use plain Excel function names (`INDEX`, `MATCH`) — do NOT use `_xlfn.` prefixes.

## 2 – Net Budget Buffer (H35:L40) and Statistics (H42:L47)

For H35:L40, each cell computes:
```
=(H19 - H12) / H26 * 100
```
where row offsets correspond: row 35↔(row 19, row 12, row 26), row 36↔(row 20, row 13, row 27), etc. Adjust the row references for each of the 6 department rows.

For H42:L47 (column-wise statistics over H35:L40):
- H42: `=MIN(H35:H40)`
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)`
- H46: `=PERCENTILE(H35:H40,0.25)`   ← Use `PERCENTILE`, NOT `PERCENTILE.INC` or `PERCENTILE.EXC`
- H47: `=PERCENTILE(H35:H40,0.75)`   ← Same

Fill across columns H–L. **Critical**: use `PERCENTILE` (not `.INC`/`.EXC`) to avoid `#NAME?` errors in Excel via openpyxl. Similarly use `MIN`, `MAX`, `MEDIAN`, `AVERAGE` without `_xlfn.` prefixes.

Verify the order (min, max, median, mean, 25th, 75th) matches the labels already in the Task sheet. Print the label cells in column D or G for rows 42–47 to confirm the ordering before writing.

## 3 – Weighted Mean (H50:L50)

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the net budget buffer percentages weighted by Approved Budget Base.

## 4 – Save

Save the workbook to `/root/output/result.xlsx`. Do NOT change any formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.

## 5 – Validation

After saving, reopen `/root/output/result.xlsx` with openpyxl (data_only=False) and print:
- A sample formula from each block (H12, H19, H26, H35, H42, H46, H47, H50) to confirm correct syntax.
- Confirm no `_xlfn.` prefixes appear in any formula.
- Confirm sheets are only `Task` and `Data`.
- Confirm the file exists and is non-empty.

If any formula contains `_xlfn.` or uses `.INC`/`.EXC` suffixes, fix it before finalizing.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=hard, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.