# Task Instruction

Execute the following steps to produce /root/output/result.xlsx.

## 0 – Preparation
```bash
mkdir -p /root/output
```
Open and inspect `/root/data/workbook.xlsx` with openpyxl to confirm:
- Sheet names (`Task`, `Data`)
- The series codes in column D on sheet `Task` (rows 12-17, 19-24, 26-31)
- The years in row 10, columns H-L
- The data layout on sheet `Data` rows 21-38 (which column holds the series code, which columns hold years, etc.)
- The labels in rows 35-40 (regions), row 42-47 (statistics labels), row 50 (GCM)
- What is already in the yellow cells (they should be empty or have placeholders)

Print all of this so you have the exact cell references before writing any formulas.

## 1 – Cross-sheet lookup formulas (Step 1)

Use `INDEX/MATCH` pattern. For each yellow cell in the three blocks (`H12:L17`, `H19:L24`, `H26:L31`), write a formula like:

```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```

Adjust the exact column/row ranges after inspecting the Data sheet layout:
- Identify which column on Data holds the series codes (could be column A or B).
- Identify which row on Data holds the year headers (could be row 20 or row 21).
- Identify the data value region accordingly.
- Use absolute references for the lookup arrays and mixed references ($D12 for the series code column, H$10 for the year row) so the formula can be filled across the block.

Write these formulas using openpyxl, setting each cell's `.value` to the formula string. Do NOT use `data_only=True` when loading.

## 2 – Net reliability gap (Step 2)

For cells `H35:L40` (six regions, five year-columns), write a formula:
```
=(H12 - H19) / H26 * 100
```
where H12 is Successful API Requests, H19 is Failed API Requests, H26 is Compute Capacity for the corresponding region and year. Use the appropriate row offsets for each of the six regions.

For the statistics block `H42:L47`:
- Row 42 (Min): `=MIN(H35:H40)` (or `=MIN(H$35:H$40)`)
- Row 43 (Max): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` — use legacy `PERCENTILE`, NOT `PERCENTILE.INC` or `PERCENTILE.EXC`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` — same, use legacy `PERCENTILE`

Verify the exact row assignments (min/max/median/mean/25th/75th) by reading the labels in column D or whatever label column is used for rows 42-47. Match the formula to the label.

## 3 – Weighted mean (Step 3)

For `H50:L50`:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net reliability gap percentages using Compute Capacity as weights.

## 4 – Save

Save the workbook to `/root/output/result.xlsx`. Do NOT change any formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.

## 5 – Validation

After saving, re-open the file with openpyxl (without data_only) and print:
- A sample formula from each block (H12, H19, H26, H35, H42-H47, H50) to confirm they are correct formula strings.
- Confirm no cells contain `#NAME?` or error literals.
- Confirm sheet names are still only `Task` and `Data`.

## Key Reminders
- Use legacy `PERCENTILE` (not `PERCENTILE.INC`/`PERCENTILE.EXC`) to avoid #NAME? errors.
- Use `INDEX/MATCH` pattern for lookups.
- Inspect the actual Data sheet layout before writing any formulas — do not assume column/row positions.
- Preserve all existing formatting.
- Mixed references are critical: lock the series-code column and the year row appropriately so formulas work when conceptually "filled" across the block.

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