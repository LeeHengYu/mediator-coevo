# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0 – Inspect
1. Copy the workbook: `cp /root/data/workbook.xlsx /root/output/result.xlsx`
2. Open `/root/output/result.xlsx` with openpyxl (data_only=False) and inspect:
   - Sheet `Task`: read row 10 (years in H10:L10), column D rows 12-17, 19-24, 26-31 (series codes), rows 35-40 (campus labels), row 50 label.
   - Sheet `Data`: read rows 21-38 — note the layout (which column holds series codes, which row holds years, and where values start). Print enough to understand the lookup geometry.
   - Print the exact cell references and their current content so you know what is blank (yellow) and what is pre-filled.

## 1 – Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For every yellow cell at row `r`, column `c` (H=8 … L=12):
- The series code is in `Task!D{r}`.
- The year is in `Task!{col_letter}10` (same column as the cell).
- The data lives on sheet `Data` rows 21:38.

Use an `INDEX(MATCH,MATCH)` pattern. Determine the exact data rectangle on `Data` (e.g., if series codes are in column A and years in row 20 with values in B21:F38, adapt accordingly). The formula template (adjust ranges after inspection) should look like:

```
=INDEX(Data!$B$21:$F$38, MATCH($D{r}, Data!$A$21:$A$38, 0), MATCH({col_letter}$10, Data!$B$20:$F$20, 0))
```

Adjust the exact ranges based on what you see during inspection. Write these formulas as strings into the cells using openpyxl.

## 2 – Step 2a: Net renewable balance (H35:L40)
The formula for each campus row `r` in 35-40 is:
```
=(H{gen_row} - H{grid_row}) / H{base_row} * 100
```
where:
- `gen_row` = corresponding row in H12:L17 (Renewable Generation block)
- `grid_row` = corresponding row in H19:L24 (Grid Consumption block)
- `base_row` = corresponding row in H26:L31 (Baseline Energy Demand block)

The campus ordering in rows 35-40 must match the campus ordering in the three blocks above. Verify this by comparing labels in column D (or whichever label column) for rows 12-17 vs 35-40. Map each campus in 35-40 to the correct row offset in the lookup blocks.

Write formulas (not values) into H35:L40.

## 2 – Step 2b: Statistics (H42:L47)
For each column `c` (H through L), write these formulas referencing H35:H40 (adjust column letter):
- Row 42 (Min):    `=MIN({c}35:{c}40)`
- Row 43 (Max):    `=MAX({c}35:{c}40)`
- Row 44 (Median): `=MEDIAN({c}35:{c}40)`
- Row 45 (Mean):   `=AVERAGE({c}35:{c}40)`
- Row 46 (25th):   `=PERCENTILE({c}35:{c}40,0.25)`
- Row 47 (75th):   `=PERCENTILE({c}35:{c}40,0.75)`

**IMPORTANT**: Use `PERCENTILE` — NOT `PERCENTILE.INC` or `PERCENTILE.EXC`. The dotted variants can produce #NAME? errors in some evaluation contexts. This was a confirmed failure mode in a sibling task.

## 3 – Step 3: Weighted mean (H50:L50)
For each column `c`:
```
=SUMPRODUCT({c}35:{c}40,{c}26:{c}31)/SUM({c}26:{c}31)
```
This uses the Net renewable balance percentages as values and Baseline Energy Demand as weights.

## 4 – Save and verify
- Save the workbook (keep_vba=False is fine; do not add macros).
- Re-open the saved file and print the formulas in a few sample cells (e.g., H12, L17, H35, H42, H46, H50) to confirm they are correctly written strings.
- Confirm no new sheets were added.
- Confirm the file exists at `/root/output/result.xlsx`.

## Key constraints
- Do NOT use data_only=True when writing; open normally.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- All formulas must be Excel formula strings (starting with '='), not computed Python values.
- Double-check row/column references after inspecting the actual workbook layout — do not assume the layout described above is pixel-perfect; adapt based on what you read.

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