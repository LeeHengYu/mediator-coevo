# Task Instruction

Create a Python script that:

1. **Inspect the workbook** at `/root/data/workbook.xlsx` using openpyxl (read-only, data_only=False) to understand the layout:
   - On sheet `Task`: read the series codes in column D for rows 12-17, 19-24, 26-31. Read the year headers in H10:L10. Read port names in column C or D for rows 35-40 if present. Note any existing content in the target cells.
   - On sheet `Data`: read the structure of rows 21-38 to understand how series codes are laid out (which column holds the code, which row holds years, where data starts).
   - Print all of this so you can verify the layout before writing formulas.

2. **Open the workbook for editing** (not read-only) and work only on the `Task` sheet.

3. **Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31:**
   For each cell in these ranges, write an `INDEX/MATCH` formula that:
   - Looks up the series code from column D of the current row against the series codes on sheet `Data` (the code column in Data rows 21:38).
   - Looks up the year from row 10 of the current column against the year headers on sheet `Data`.
   - Uses the pattern: `=INDEX(Data!<data_range>,MATCH(<Task_D_ref>,Data!<code_column>,0),MATCH(<Task_year_ref>,Data!<year_header_range>,0))`
   - Make sure the Data range references use absolute anchoring ($) for the lookup arrays but relative references for the current row's D column and current column's row-10 year.
   - Every formula string must start with '='.

4. **Step 2 – Net container flow in H35:L40:**
   For each cell, write a formula: `=(H12-H19)/H26*100` adjusting row references for the correct port (row 12 maps to row 35, row 13 to 36, etc.; row 19 maps to row 35's outbound, etc.; row 26 maps to row 35's capacity, etc.). Specifically:
   - Row 35: `=(<col>12-<col>19)/<col>26*100`
   - Row 36: `=(<col>13-<col>20)/<col>27*100`
   - Row 37: `=(<col>14-<col>21)/<col>28*100`
   - Row 38: `=(<col>15-<col>22)/<col>29*100`
   - Row 39: `=(<col>16-<col>23)/<col>30*100`
   - Row 40: `=(<col>17-<col>24)/<col>31*100`

5. **Step 2 – Summary statistics in H42:L47:**
   For each column (H through L):
   - Row 42: `=MIN(<col>35:<col>40)`
   - Row 43: `=MAX(<col>35:<col>40)`
   - Row 44: `=MEDIAN(<col>35:<col>40)`
   - Row 45: `=AVERAGE(<col>35:<col>40)`
   - Row 46: `=PERCENTILE(<col>35:<col>40,0.25)` — use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) for maximum compatibility
   - Row 47: `=PERCENTILE(<col>35:<col>40,0.75)`

6. **Step 3 – Weighted mean in H50:L50:**
   For each column (H through L):
   `=SUMPRODUCT(<col>35:<col>40,<col>26:<col>31)/SUM(<col>26:<col>31)`

7. **Save** the workbook to `/root/output/result.xlsx` (create `/root/output/` directory if needed). Do NOT use `data_only=True` when opening for editing.

8. **Verify** by reopening the saved file and printing the formula content of a sample of cells (e.g., H12, L17, H35, L40, H42, H47, H50) to confirm formulas were written correctly.

**Critical details:**
- All formula strings must begin with `=`.
- Use `PERCENTILE` (not `PERCENTILE.INC`) for rows 46-47.
- Do not modify any existing formatting, sheets, or content outside the specified cells.
- Ensure you write to the correct sheet object (`ws = wb['Task']`).
- After the inspection step, print the discovered layout so you can adapt the exact Data range references (row/column for codes, row for years, data area) before writing formulas.

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