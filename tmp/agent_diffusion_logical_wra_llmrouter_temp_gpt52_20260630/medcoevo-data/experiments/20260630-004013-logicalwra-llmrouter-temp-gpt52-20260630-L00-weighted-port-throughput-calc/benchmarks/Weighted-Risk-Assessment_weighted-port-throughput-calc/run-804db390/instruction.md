# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl and inspect:
- Sheet names (confirm `Task` and `Data` exist)
- On sheet `Task`: read cells D12:D17, D19:D24, D26:D31 to get the series codes for each block; read H10:L10 to get the year headers; read D35:D40 for port names; read H42:H47 labels (min, max, median, mean, 25th, 75th percentile); read D50 for the CPA label.
- On sheet `Data`: read row 21 headers and rows 21:38 to understand the data layout (which row is header, which column has series codes, which columns/rows have year data).
- Print all of this so you understand the exact layout before writing any formulas.

## 2. Understand the Data sheet layout
Determine:
- Whether Data rows 21:38 are arranged with series codes in a column and years across columns (suitable for VLOOKUP/INDEX-MATCH), or series codes in a row and years down rows (suitable for HLOOKUP).
- The exact column letters and row numbers for the data range on the Data sheet.
- Whether the year values in Data match the years in Task row 10.

## 3. Write lookup formulas in H12:L17, H19:L24, H26:L31
Using openpyxl, write Excel formulas (as strings starting with `=`) into each cell. Use INDEX-MATCH pattern since it's the most flexible.

For each cell at row `r`, column `c` (where c maps to H=8, I=9, J=10, K=11, L=12):
- The series code is in cell `$D{r}` on sheet Task
- The year is in cell `{col_letter}$10` on sheet Task (e.g., H$10, I$10, etc.)
- The data is on sheet Data in a specific range

The formula pattern should be something like:
`=INDEX(Data!<data_values_range>, MATCH($D{r}, Data!<series_code_column>, 0), MATCH({col}$10, Data!<year_header_row>, 0))`

Adjust the exact ranges based on what you discovered in step 1-2. The data range in the INDEX should cover the numeric values only; the MATCH for the series code should reference the column containing series codes; the MATCH for the year should reference the row containing year headers.

IMPORTANT: Use absolute references where appropriate ($D for the series code column, $10 for the year row) so formulas are consistent. Make sure to use the correct sheet reference syntax: `Data!` prefix.

Write these formulas for all 3 blocks (rows 12-17, 19-24, 26-31), columns H through L.

## 4. Write Net Container Flow formulas in H35:L40
For each cell at row `r` in 35:40 and column `c` in H:L:
- Loaded Containers Inbound is in the first block (H12:L17) — determine which row offset corresponds to this port
- Loaded Containers Outbound is in the second block (H19:L24)
- Terminal Throughput Capacity is in the third block (H26:L31)

The mapping: row 35 corresponds to the first port (rows 12, 19, 26), row 36 to second (rows 13, 20, 27), etc.

Formula: `=({col}{inbound_row} - {col}{outbound_row}) / {col}{capacity_row} * 100`

For example, H35: `=(H12-H19)/H26*100`

## 5. Write summary statistics in H42:L47
For each column c in H:L, write these formulas:
- Row 42 (minimum): `=MIN({c}35:{c}40)`
- Row 43 (maximum): `=MAX({c}35:{c}40)`
- Row 44 (median): `=MEDIAN({c}35:{c}40)`
- Row 45 (mean): `=AVERAGE({c}35:{c}40)`
- Row 46 (25th percentile): `=PERCENTILE({c}35:{c}40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE({c}35:{c}40, 0.75)`

IMPORTANT: Check the labels in column D/G for rows 42-47 to confirm the correct order of statistics. Adjust the row assignments if the labels differ from what's assumed above.

## 6. Write weighted mean in H50:L50
For each column c in H:L:
`=SUMPRODUCT({c}35:{c}40, {c}26:{c}31) / SUM({c}26:{c}31)`

This computes the weighted mean of the net container flow percentages (H35:L40) weighted by terminal throughput capacity (H26:L31).

## 7. Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets.

## 8. Verify
Re-open `/root/output/result.xlsx` and print:
- A sample of formulas from each block (H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50)
- Confirm all target cells contain formula strings (not None/empty)
- Confirm no new sheets were added
- Confirm the file is valid xlsx

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