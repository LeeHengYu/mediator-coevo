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
- On sheet `Task`: read row 10 (the year headers in columns H–L), read column D rows 12–17, 19–24, 26–31 (the series codes), read row labels for rows 12–17, 19–24, 26–31 to understand what each block represents (Renewable Generation, Grid Consumption, Baseline Energy Demand or similar)
- On sheet `Data`: read rows 21–38 to understand the data layout — identify which column holds series codes, which row holds years, and the data orientation (is it a vertical table with series codes in one column and years across columns, or something else?)
- Read rows 35–40 on `Task` sheet to see the campus names and any existing content
- Read rows 42–47 labels (min, max, median, mean, 25th percentile, 75th percentile)
- Read row 50 label
- Print all findings so you can construct correct formulas.

## 2. Determine exact Data sheet layout
From step 1 output, determine:
- The column on `Data` sheet that contains the series/lookup codes (likely column A or B)
- The row on `Data` sheet that contains the year values that match row 10 on `Task` sheet
- The range of data columns
This is critical for building correct VLOOKUP/INDEX-MATCH formulas.

## 3. Write formulas using openpyxl
Use a Python script with openpyxl to write formulas into the cells. Load the workbook with `data_only=False` to preserve existing formulas. Key points:

### Step 1 formulas (H12:L17, H19:L24, H26:L31)
For each cell in these ranges, write an INDEX-MATCH formula. The formula pattern should be:
```
=INDEX(Data!<data_range>, MATCH($D{row}, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```
Adjust the exact references based on what you discovered in the inspection:
- `$D{row}` — the series code in column D of the current row on Task sheet (use $ to lock column)
- `H$10` (or I$10, J$10, etc.) — the year from row 10 (use $ to lock row)
- `Data!<data_range>` — the rectangular block of numeric data on the Data sheet (rows 21:38, appropriate columns)
- `Data!<series_code_column>` — the column of series codes on Data sheet
- `Data!<year_row>` — the row of years on Data sheet

Make sure the INDEX range, the MATCH lookup column, and the MATCH lookup row are all consistent and correctly sized.

### Step 2 formulas (H35:L40) — Net renewable balance
Assuming rows 12–17 are Renewable Generation, rows 19–24 are Grid Consumption, and rows 26–31 are Baseline Energy Demand (verify from inspection), for each campus i (0–5) and each year column col (H–L):
```
=({col}{12+i} - {col}{19+i}) / {col}{26+i} * 100
```
So H35 = `=(H12-H19)/H26*100`, H36 = `=(H13-H20)/H27*100`, etc.

Adjust the row references if the blocks don't map exactly as assumed — use the actual row numbers discovered in inspection.

### Step 2 statistics (H42:L47)
For each column col in H–L:
- H42 (min): `=MIN(H35:H40)`
- H43 (max): `=MAX(H35:H40)`
- H44 (median): `=MEDIAN(H35:H40)`
- H45 (mean): `=AVERAGE(H35:H40)`
- H46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)`
- H47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)`

Verify the order of statistics by reading the row labels from the inspection. The labels might be in a different order — match each formula to its label.

### Step 3 formula (H50:L50) — Weighted mean
For each column col in H–L:
```
=SUMPRODUCT({col}35:{col}40, {col}26:{col}31) / SUM({col}26:{col}31)
```

## 4. Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change any formatting, do NOT add sheets.

## 5. Validate
Reopen the saved file with openpyxl and confirm:
- All target cells contain formula strings (not None or literal values)
- The formulas reference correct sheets and ranges
- No extra sheets were added
- Print a sample of formulas from each block for verification

## Important constraints
- Use `openpyxl` only, no xlsxwriter
- Load with `data_only=False` 
- Do not modify any cells outside the specified ranges
- Do not add sheets, macros, VBA, external links, or helper tabs
- Preserve all existing formatting

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