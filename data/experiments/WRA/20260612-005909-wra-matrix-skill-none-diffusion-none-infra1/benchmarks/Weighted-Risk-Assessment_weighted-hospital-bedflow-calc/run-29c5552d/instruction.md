# Task Instruction

Execute the following steps in a single Python script.

## Phase 1 — Inspect the workbook structure

1. Open `/root/data/workbook.xlsx` with `openpyxl` (default `data_only=False`).
2. Print the sheet names.
3. For the `Task` sheet:
   - Print rows 10–50, columns A–L (values and any existing formulas). Pay special attention to:
     - Row 10 (the year headers in H10:L10)
     - Column D rows 12–31 (the series codes)
     - The labels in column A or B for rows 12–17, 19–24, 26–31 (the three blocks)
     - Rows 35–40 (hospital names / labels for Net patient flow)
     - Rows 42–47 (stat labels: min, max, median, mean, 25th, 75th percentile)
     - Row 50 (weighted mean row)
4. For the `Data` sheet:
   - Print rows 1–5 to see headers / structure.
   - Print rows 21–38 fully (all columns) to see the source data layout.
   - Identify: which row contains headers (series codes? years?), which column holds series codes, which columns/rows hold years.

Print everything clearly with row numbers and column letters. Do NOT write anything yet.

## Phase 2 — Write formulas (after inspecting output from Phase 1)

Based on the inspection, write a second section of the script that:

### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three 6×5 blocks, write an `INDEX(MATCH,MATCH)` formula that:
- Looks up the series code from column D of the current row (use `$D` + row for the row-absolute reference on the series code).
- Looks up the year from row 10 of the current column (use column letter + `$10` for the column-absolute reference on the year).
- References the data area and header arrays on the `Data` sheet rows 21–38.
- Use the exact column range for series codes and the exact row range for years as found in the inspection.
- The formula pattern should be like: `=INDEX(Data!$B$22:$Z$38,MATCH($D12,Data!$A$22:$A$38,0),MATCH(H$10,Data!$B$21:$Z$21,0))` — but adjust all references to match the actual layout discovered in Phase 1.

IMPORTANT: Make sure the INDEX data range, the MATCH lookup arrays, and the cell references are exactly correct based on the actual workbook structure.

### Step 2: Net patient flow in H35:L40

For each of the 6 hospitals (rows 35–40) and 5 year columns (H–L):
- Identify which rows in the three blocks correspond to Patient Admissions, Patient Discharges, and Effective Bed Capacity for each hospital. The three blocks (rows 12–17, 19–24, 26–31) each contain 6 hospitals in the same order as rows 35–40.
- Write a formula: `=(H12-H19)/H26*100` pattern, adjusting row references for each hospital. For hospital index i (0–5): admissions row = 12+i, discharges row = 19+i, capacity row = 26+i, result row = 35+i.

### Step 3: Summary statistics in H42:L47

For each column H–L:
- Row 42: `=MIN(H35:H40)` (or whichever row is MIN based on labels)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

Match the stat function to the label in column A/B/C of that row. Print the labels first to confirm the order.

### Step 4: Weighted mean in H50:L50

For each column H–L:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

### Step 5: Save

Save to `/root/output/result.xlsx`. Create the `/root/output/` directory if needed.

## Critical Notes
- Do NOT add sheets, macros, VBA, or helper columns.
- Do NOT change any existing formatting.
- Run the full inspection FIRST, print results, then construct formulas based on what you actually see.
- After writing formulas, re-read a few cells (e.g., H12, H19, H26, H35, H42, H50) to confirm the formula strings were written.
- If any column/row mapping is ambiguous, print more of the Data sheet to resolve it before writing.

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