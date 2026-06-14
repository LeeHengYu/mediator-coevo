# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
- Open `/root/data/workbook.xlsx` with openpyxl (data_only=False to preserve formulas).
- Print the sheet names to confirm `Task` and `Data` exist.
- Print the contents of `Task` sheet rows 10-50, columns D through L, to understand the layout: what's in D12:D17, D19:D24, D26:D31 (series codes), row 10 H-L (years), and the labels in rows 35-50.
- Print `Data` sheet rows 21-38 fully (all columns) to understand the data layout: which column has series codes, which row has years, and how data is arranged.
- Print `Task` sheet rows 42-47 column C or D area to see what statistics labels are (min, max, median, mean, 25th, 75th percentile).
- Print `Task` sheet row 50 to see the MHN weighted mean row.

## 2. Determine the lookup structure
Based on inspection of the Data sheet rows 21-38:
- Identify which column contains the series codes (likely column A or B on Data sheet).
- Identify which row contains the year headers.
- Determine the exact range for VLOOKUP/INDEX/MATCH formulas.

## 3. Step 1 — Populate H12:L17, H19:L24, H26:L31 with lookup formulas
For each cell in these ranges, write a formula that:
- Uses the series code from column D of that row on the Task sheet
- Uses the year from row 10 of that column on the Task sheet
- Looks up the value from Data sheet rows 21:38

Use INDEX/MATCH/MATCH or VLOOKUP with MATCH pattern. The exact formula depends on the Data sheet layout discovered in step 1. Example pattern (adjust based on actual layout):
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
Adjust column/row references based on actual data layout. Use absolute references for the data range and mixed references ($D12 for row-lock on column, H$10 for column-lock on row) so formulas can be applied across the grid.

Write these formulas using openpyxl by setting cell.value to the formula string (without leading equals sign in openpyxl — actually openpyxl DOES use the leading = sign). Set each cell's value to the formula string like `'=INDEX(Data!...)'`.

## 4. Step 2 — Net patient flow in H35:L40
Based on the layout:
- H12:L17 likely = Patient Admissions (or one of the three blocks)
- H19:L24 likely = Patient Discharges
- H26:L31 likely = Effective Bed Capacity

Verify which block is which by checking the labels. Then for each cell in H35:L40:
```
= (Admissions_cell - Discharges_cell) / Capacity_cell * 100
```
For example, if admissions are in rows 12-17, discharges in 19-24, capacity in 26-31:
```
H35 = (H12 - H19) / H26 * 100
```
Adjust row offsets for each of the 6 hospitals.

## 5. Step 2 — Statistics in H42:L47
For each column H through L, calculate these 6 statistics over the 6 net-flow values (H35:H40 for column H, etc.):
- Row 42: MIN — `=MIN(H35:H40)`
- Row 43: MAX — `=MAX(H35:H40)`
- Row 44: MEDIAN — `=MEDIAN(H35:H40)`
- Row 45: AVERAGE — `=AVERAGE(H35:H40)`
- Row 46: PERCENTILE — `=PERCENTILE(H35:H40, 0.25)`
- Row 47: PERCENTILE — `=PERCENTILE(H35:H40, 0.75)`

**IMPORTANT**: Check the actual labels in column C/D for rows 42-47 to determine the correct order. The order above is a guess — match the actual labels.

## 6. Step 3 — Weighted mean in H50:L50
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the net patient flow percentages as values and Effective Bed Capacity as weights.

## 7. Save
Save the workbook to `/root/output/result.xlsx` using openpyxl. Do NOT use data_only mode — preserve all formulas.

## 8. Verify
- Reopen the saved file and print cells from each formula region to confirm formulas are present (not None or empty).
- Confirm no new sheets were added.
- Confirm the formulas reference the correct ranges.

## Critical Notes
- Do NOT add any new sheets, macros, or VBA.
- Do NOT change existing formatting — only set cell values (formulas).
- Use openpyxl throughout. Load with `data_only=False`.
- When writing formulas, include the `=` sign at the start of the string.
- Adapt all cell references based on what you actually observe in the workbook — do not blindly use the example references above if the data layout differs.
- For PERCENTILE function: use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) unless you determine the verifier expects a specific variant. `PERCENTILE` is safest for broad compatibility.
- Double-check that the series code column and year header row on the Data sheet are correctly identified before writing any formulas.

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