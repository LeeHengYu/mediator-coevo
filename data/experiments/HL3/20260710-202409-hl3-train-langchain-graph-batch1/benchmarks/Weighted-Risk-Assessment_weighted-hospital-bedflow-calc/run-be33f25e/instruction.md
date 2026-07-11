# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 – Inspect the workbook
```bash
mkdir -p /root/output
```
Open and inspect `/root/data/workbook.xlsx` with openpyxl (read-only first). Print:
- Sheet names.
- `Task` sheet: contents of rows 10-50, columns D-L (values + any existing formulas). Pay special attention to:
  - Row 10 (years in H10:L10).
  - Column D rows 12-31 (series codes).
  - The structure of rows 35-50.
- `Data` sheet: rows 21-38, all columns. Identify the layout: which column holds the series code, which row/column holds years, and where the values are.

Print everything so you can design correct formulas.

## 1 – Write the formulas with openpyxl

Use `openpyxl` to open the workbook (NOT data_only), write formulas, and save.

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in those ranges, write an INDEX/MATCH formula that:
- Matches the series code in column D of that row against the series-code column on `Data` (rows 21-38).
- Matches the year in row 10 of the same column against the year row on `Data`.
- Returns the intersecting value.

Use this pattern (adjust column/row references after inspection):
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12,Data!$A$21:$A$38,0), MATCH(H$10,Data!$B$20:$Z$20,0))
```
Adjust `$B$21:$Z$38`, `$A$21:$A$38`, `$B$20:$Z$20` to the actual data layout you discover in step 0. The series-code column anchor (`$D12`) must use a mixed reference so it locks the column but not the row. The year reference (`H$10`) must lock the row but not the column.

### Step 2 – Net patient flow (H35:L40)

For each hospital (rows 35-40) and each year column (H-L), write:
```
=(H12-H19)/H26*100
```
where H12 is the corresponding Admissions cell, H19 is Discharges, H26 is Effective Bed Capacity. Adjust row references per hospital (hospital 1 → rows 12,19,26; hospital 2 → rows 13,20,27; etc.).

### Step 2 – Summary statistics (H42:L47)

For each year column (H-L):
- Row 42 (Min):    `=MIN(H35:H40)`
- Row 43 (Max):    `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean):   `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`). The `.INC`/`.EXC` variants cause `#NAME?` errors in some evaluation environments. Similarly use `MEDIAN`, `MIN`, `MAX`, `AVERAGE` — all universally supported.

### Step 3 – Weighted mean (H50:L50)

For each year column (H-L):
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean of the net-patient-flow percentages using Effective Bed Capacity as weights.

## 2 – Save

Save to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets.

## 3 – Validate

Re-open the saved file and print all formula cells you wrote to confirm:
- Every cell in H12:L31 contains an INDEX/MATCH formula.
- Every cell in H35:L40 contains the net-flow formula.
- Every cell in H42:L47 contains the correct stats formula.
- Every cell in H50:L50 contains the SUMPRODUCT formula.
- No cells contain `#NAME?` or other error literals.
- No new sheets were added.

If any formula looks wrong, fix it before finishing.

## 4 – Run verifier if available

Check if `/root/tests/` or similar test directory exists. If so, run `python -m pytest /root/tests/ -v` and report results. Fix any failures.

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