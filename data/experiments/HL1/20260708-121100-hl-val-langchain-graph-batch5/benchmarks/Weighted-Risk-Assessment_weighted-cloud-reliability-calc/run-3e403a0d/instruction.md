# Task Instruction

Execute the following steps to complete the task.

## 0 – Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1 – Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Examine:
- Sheet names (confirm `Task` and `Data` exist).
- On `Task`: read column D rows 12-17, 19-24, 26-31 to see the series codes; read row 10 columns H-L to see the years; read rows 35-40 (labels/structure), rows 42-47 (stat labels), and row 50 (GCM label). Print all of these.
- On `Data`: read rows 21-38 to understand the layout (which row holds headers, which column holds series codes, which columns hold year data). Print the first few columns and the header row so you know the exact structure.

Do NOT proceed until you have printed and understood the layout.

## 2 – Populate H12:L17, H19:L24, H26:L31 with lookup formulas
For each block of 6 rows × 5 columns, write a formula into each yellow cell. The formula must combine:
- The series code from column D of that row (e.g., `$D12`)
- The year from row 10 of that column (e.g., `H$10`)
- A lookup into sheet `Data` rows 21:38

Use INDEX/MATCH (nested). The pattern should be:
```
=INDEX(Data!<data_area>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```
Determine the exact ranges from your inspection in step 1:
- `<data_area>`: the rectangular block on Data that contains the numeric values (likely starting from the first data column to the last, rows 21-38).
- `<series_code_column>`: the column on Data that holds the series codes (same rows 21-38).
- `<year_header_row>`: the row on Data that holds the year headers (same columns as the data area).

Make sure references use appropriate absolute/relative anchoring:
- Column D reference: `$D12` (column absolute, row relative)
- Row 10 reference: `H$10` (column relative, row absolute)
- Data ranges: fully absolute with `$`

Write these formulas using openpyxl by assigning formula strings to each cell.

## 3 – Populate H35:L40 with Net reliability gap
The formula for each cell is:
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to Successful API Requests, H19 to Failed API Requests, H26 to Compute Capacity, adjusted for the correct row offsets within each block. Specifically:
- Row 35 uses data from rows 12, 19, 26
- Row 36 uses data from rows 13, 20, 27
- Row 37 uses data from rows 14, 21, 28
- Row 38 uses data from rows 15, 22, 29
- Row 39 uses data from rows 16, 23, 30
- Row 40 uses data from rows 17, 24, 31

For column H through L, write e.g. for cell H35:
`=(H12-H19)/H26*100`

## 4 – Populate H42:L47 with summary statistics
For each column (H through L), compute over the range of 6 values in rows 35-40:
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

Verify the row labels match (print them). If the order differs, match formulas to labels.

## 5 – Populate H50:L50 with weighted mean (SUMPRODUCT)
For each column (H through L):
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This computes the weighted mean of the Net reliability gap values using Compute Capacity as weights.

## 6 – Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets.

## 7 – Verify
Reopen the saved file and print a sample of the formula cells (e.g., H12, H19, H26, H35, H42, H50) to confirm formulas were written correctly. Check that no extra sheets were added and formatting is intact.

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