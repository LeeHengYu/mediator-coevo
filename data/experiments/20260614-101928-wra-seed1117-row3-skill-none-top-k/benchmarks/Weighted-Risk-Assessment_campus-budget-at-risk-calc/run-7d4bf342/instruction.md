# Task Instruction

Execute the following steps exactly:

1. **Inspect the workbook** 
 ```bash
 cp /root/data/workbook.xlsx /root/output/result.xlsx
 ```
 Then open `/root/output/result.xlsx` with openpyxl (data_only=False) and print:
 - Sheet names.
 - On sheet `Task`: the contents of column D rows 12-17, 19-24, 26-31 (series codes), row 10 columns H-L (years), and any existing content in H12:L31, H35:L40, H42:L47, H50:L50.
 - On sheet `Data`: rows 21-38 fully (all columns) so we can see the layout — column headers, series codes, and where the year data lives.

2. **Populate H12:L17, H19:L24, H26:L31 with INDEX/MATCH formulas** 
 For each yellow cell at row `r`, column `c` (H=8 … L=12):
 - The series code is in `Task!D{r}` (the same row).
 - The year is in `Task!{col_letter}10` (the same column, row 10).
 - The data lives on sheet `Data` rows 21-38.
 - Determine from your inspection which column on `Data` holds the series codes and which row holds the year headers. Then write an INDEX/MATCH/MATCH or INDEX(MATCH,MATCH) formula referencing `Data!` with absolute row/column references for the data block and relative references for the two criteria.
 - Use the pattern: 
 ```
 =INDEX(Data!$B$21:$Z$38, MATCH(D{r}, Data!$A$21:$A$38, 0), MATCH({col}10, Data!$B$20:$Z$20, 0))
 ```
 Adjust column letters (`$A`, `$B`, `$Z`, row `$20`) based on what you actually see in the Data sheet. The key: the first MATCH finds the series code row; the second MATCH finds the year column.

3. **H35:L40 — Net budget buffer** 
 For each department row `r` in 35-40 and each year column `c` in H-L:
 - Committed Funding is in the H19:L24 block (same relative position: row offset = r-35 maps to 19+offset, col is same).
 - Operating Spend is in the H12:L17 block (same relative position: row offset maps to 12+offset).
 - Approved Budget Base is in the H26:L31 block (same relative position: row offset maps to 26+offset).
 - Formula: `=({col}{19+offset} - {col}{12+offset}) / {col}{26+offset} * 100`
 - Example for H35: `=(H19-H12)/H26*100`

4. **H42:L47 — Column-wise statistics** 
 For each year column `c` (H-L):
 - Row 42 (MIN): `=MIN({c}35:{c}40)`
 - Row 43 (MAX): `=MAX({c}35:{c}40)`
 - Row 44 (MEDIAN): `=MEDIAN({c}35:{c}40)`
 - Row 45 (MEAN): `=AVERAGE({c}35:{c}40)`
 - Row 46 (25th percentile): `=PERCENTILE({c}35:{c}40,0.25)` — use `PERCENTILE`, NOT `PERCENTILE.INC`
 - Row 47 (75th percentile): `=PERCENTILE({c}35:{c}40,0.75)` — use `PERCENTILE`, NOT `PERCENTILE.EXC`

5. **H50:L50 — Weighted mean** 
 For each year column `c` (H-L):
 - `=SUMPRODUCT({c}35:{c}40,{c}26:{c}31)/SUM({c}26:{c}31)`

6. **Save** 
 Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets, macros, or VBA.

7. **Validate** 
 - Re-open the saved file with openpyxl (data_only=False).
 - Print the formulas in a sample of cells (e.g., H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50) to confirm they are present and correctly structured.
 - Ensure no cells contain `#NAME?` literal strings (which would indicate formula name issues).
 - Run any test suite if present: `cd /root && python -m pytest tests/ -x -v 2>&1 | head -80`

**Critical reminders:**
- Use `PERCENTILE` (legacy name), never `PERCENTILE.INC` or `PERCENTILE.EXC`.
- Use absolute references for the Data sheet lookup ranges and relative references for the two lookup criteria.
- Verify the actual Data sheet layout before writing formulas — do not assume column/row positions.
- Confirm row 10 on Task has the year values and column D has the series codes by inspection.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=hard, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.