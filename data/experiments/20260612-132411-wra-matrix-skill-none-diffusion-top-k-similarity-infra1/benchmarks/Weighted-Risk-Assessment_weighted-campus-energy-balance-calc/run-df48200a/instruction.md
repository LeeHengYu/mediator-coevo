# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 – Environment & Inspection
```bash
mkdir -p /root/output
pip install openpyxl
```
Open and inspect `/root/data/workbook.xlsx` with openpyxl to understand:
- Sheet names (expect `Task` and `Data`).
- On `Task`: read column D rows 12-17, 19-24, 26-31 to see the series codes; read row 10 columns H-L to see the years; read rows 35-40 for campus labels; read row 50 for MCEC; read rows 42-47 for stat labels.
- On `Data`: read rows 21-38 to understand the layout (which column holds the series code, which row holds years, where the numeric data starts).
Print all of this so you know the exact structure before writing any formulas.

## 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31

For every cell in these three blocks, write a formula that looks up the value from `Data!$21:$38`. The two inputs are:
- The series code in column D of the current row on `Task`.
- The year in row 10 of the current column on `Task`.

Use INDEX/MATCH. The exact pattern depends on how Data is laid out. Typical pattern if Data has series codes in a column (say column A) and years in a header row (say row 21):
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```
Adjust the ranges after inspecting the actual layout. Lock references appropriately ($D12 for the code column, H$10 for the year row) so the formula can be filled across the block.

IMPORTANT: When writing formulas with openpyxl, assign the formula string directly to `cell.value`. Do NOT use `data_only` mode for writing. Make sure every formula string starts with `=`.

## 2 – Net renewable balance in H35:L40

For each campus (rows 35-40) and each year (columns H-L), write:
```
=(H12 - H19) / H26 * 100
```
where H12 is the Renewable Generation row for that campus, H19 is Grid Consumption, and H26 is Baseline Energy Demand. Adjust row references to match the correct campus within each block. For example, if row 35 corresponds to the first campus:
- Renewable Generation = row 12
- Grid Consumption = row 19  
- Baseline Energy Demand = row 26

Row 36 → rows 13, 20, 27; etc.

## 3 – Summary statistics in H42:L47

For each year column (H through L), write column-wise formulas over the net-balance block (rows 35:40):
- Row 42 (Min): `=MIN(H35:H40)`
- Row 43 (Max): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

CRITICAL: Use `PERCENTILE` (legacy name), NOT `PERCENTILE.INC`. The dot-suffixed version causes #NAME? errors in this environment. Similarly use `MEDIAN`, `MIN`, `MAX`, `AVERAGE` (all legacy names).

## 4 – Weighted mean in H50:L50

For each year column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the net renewable balance percentages using Baseline Energy Demand as weights.

## 5 – Save

Save the workbook to `/root/output/result.xlsx`. Do NOT change any existing formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.

## 6 – Verification

After saving, reopen the file and spot-check:
- That cells in H12:L17, H19:L24, H26:L31 contain formula strings (not None).
- That cells in H35:L40 contain formula strings.
- That cells in H42:L47 contain formula strings.
- That cells in H50:L50 contain formula strings.
- Print a sample of these to confirm.

If any cell is None or empty, debug immediately by re-reading the workbook structure and fixing the formulas before re-saving.

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