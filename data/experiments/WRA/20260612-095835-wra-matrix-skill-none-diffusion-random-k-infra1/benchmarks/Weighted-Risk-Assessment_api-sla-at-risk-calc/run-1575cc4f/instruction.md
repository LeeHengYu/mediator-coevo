# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
cp /root/data/workbook.xlsx /root/output/result.xlsx
```

## 1. Inspect the workbook structure
Open `/root/output/result.xlsx` with openpyxl and inspect:
- Sheet names (confirm `Task` and `Data` exist)
- On sheet `Task`: read row 10 (especially H10:L10) to see the year headers; read column D rows 12-17, 19-24, 26-31 to see the series codes; read rows 35-40 labels; read row 50 label; read rows 42-47 labels (min, max, median, mean, 25th, 75th percentile).
- On sheet `Data`: read rows 21-38 to understand the data layout — identify which row is the header row, which column has series codes, and how the data is arranged (rows vs columns).
- Check what's in the yellow cells already (H12:L17 etc.) — they should be empty or placeholder.
- Print all findings clearly before proceeding.

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each yellow cell at row `r` and column `c` (H=8, I=9, J=10, K=11, L=12):
- The series code is in cell D of the same row `r` (e.g., `$D12`)
- The year is in row 10 of the same column `c` (e.g., `H$10`)
- The data source is `Data!$A$21:$Z$38` (adjust column range based on inspection — use the actual extent of data found)

Use the INDEX/MATCH pattern:
```
=INDEX(Data!<data_range>, MATCH($D<r>, Data!<series_code_column>, 0), MATCH(<c>$10, Data!<year_header_row>, 0))
```

IMPORTANT: Based on the inspection of the Data sheet:
- Identify which column contains the series codes (likely column A or B of Data sheet)
- Identify which row contains the year headers in the data range
- Construct the INDEX range, the MATCH lookup arrays accordingly
- Use absolute references for the data range and lookup arrays, mixed references for the varying inputs ($D<r> for series code, <col>$10 for year)

Write these formulas using openpyxl. Since openpyxl doesn't evaluate formulas, just set cell.value to the formula string (starting with `=`).

The three blocks H12:L17, H19:L24, H26:L31 each have 6 rows × 5 columns = 30 formulas each, 90 total.

## 3. Net SLA Buffer formulas in H35:L40

For each cell at row `r_out` (35-40) and column `c` (H-L):
- The corresponding row offset `i` is 0-5 (service index)
- Latency Budget Preserved is in the first block: row 12+i, same column (H12:L17)
- Latency Budget Consumed is in the second block: row 19+i, same column (H19:L24)  
- Covered Request Capacity is in the third block: row 26+i, same column (H26:L31)

Formula: `=(<Preserved_cell> - <Consumed_cell>) / <Capacity_cell> * 100`

For row 35, column H: `=(H12-H19)/H26*100`
For row 35, column I: `=(I12-I19)/I26*100`
etc.

Verify by checking labels in column D of rows 35-40 match those in rows 12-17, 19-24, 26-31. If the service order differs, adjust the row references accordingly.

## 4. Summary statistics in H42:L47

For each column `c` (H through L), the six stats reference the Net SLA buffer values in rows 35:40 of that column. Based on the labels found in rows 42-47 during inspection, assign:
- Minimum: `=MIN(<c>35:<c>40)`
- Maximum: `=MAX(<c>35:<c>40)`
- Median: `=MEDIAN(<c>35:<c>40)`
- Mean: `=AVERAGE(<c>35:<c>40)`
- 25th percentile: `=PERCENTILE(<c>35:<c>40,0.25)` (or PERCENTILE.INC)
- 75th percentile: `=PERCENTILE(<c>35:<c>40,0.75)` (or PERCENTILE.INC)

Match each formula to the correct row based on the actual labels found. Use PERCENTILE (not PERCENTILE.EXC) unless inspection suggests otherwise.

## 5. Weighted mean in H50:L50

For each column `c`: `=SUMPRODUCT(<c>35:<c>40, <c>26:<c>31) / SUM(<c>26:<c>31)`

This computes the weighted mean of Net SLA buffer percentages weighted by Covered Request Capacity.

## 6. Save and verify
- Save the workbook.
- Reopen it and print out all formula cells to verify they are correctly placed.
- Confirm no new sheets were added, no macros, no external links.
- Confirm the file is saved at `/root/output/result.xlsx`.

## Critical Notes
- Do NOT use data_only=True when opening for writing — you need to preserve and write formulas.
- Do NOT delete or modify any existing content outside the specified cells.
- Do NOT change formatting. When writing formulas, only set cell.value.
- If the inspection in step 1 reveals a different layout than assumed (e.g., different row/column for series codes on Data sheet, different year header row), adapt all formulas accordingly. The inspection step is critical — do not skip it.
- Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC` for maximum compatibility unless the workbook already uses a specific variant.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=medium, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.