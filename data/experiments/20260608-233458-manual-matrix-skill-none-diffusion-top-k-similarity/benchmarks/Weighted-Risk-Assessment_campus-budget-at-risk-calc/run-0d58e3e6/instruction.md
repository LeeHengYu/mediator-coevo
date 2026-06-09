# Task Instruction

Execute the following steps in order.

## Phase 0 — Inspect the workbook
1. `pip install openpyxl` if needed.
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False so you see formulas).
3. Print the sheet names.
4. On sheet **Task**:
   - Print rows 10-11 (to see the year headers in H10:L10).
   - Print rows 12-31 column D (to see the series codes for each block).
   - Print rows 35-50 to understand the layout of the Net budget buffer block, summary stats block, and weighted-mean row.
   - Print the exact text labels in cells A42:A47 (or G42:G47) so you know which row is min, max, median, mean, 25th pctl, 75th pctl.
5. On sheet **Data**:
   - Print rows 21-38 fully (all columns with data) to see the source data layout — determine whether series codes are in a row or column, and where years appear.
   - Note the exact orientation: are years in a row header and series codes in a column, or vice-versa?

Record all findings before writing any formulas.

## Phase 1 — Lookup formulas in H12:L31
Using INDEX-MATCH (preferred) or another allowed pattern:
- For each cell in H12:L17, H19:L24, H26:L31, write a formula that looks up:
  - The series code from column D of that row on sheet Task
  - The year from row 10 of that column on sheet Task
  - Against the data in sheet Data rows 21:38
- Use absolute references for the Data range and MATCH ranges so the formula is correct across the block.
- Example pattern (adapt after inspecting orientation):
  `=INDEX(Data!$B$21:$XX$38, MATCH($D12,Data!$A$21:$A$38,0), MATCH(H$10,Data!$B$20:$XX$20,0))`
  Adjust column/row letters based on actual layout discovered in Phase 0.

## Phase 2 — Net budget buffer (H35:L40)
The three input blocks are:
- Committed Funding: H12:L17
- Operating Spend: H19:L24
- Approved Budget Base: H26:L31

Formula for each cell: `=(H12-H19)/H26*100` (adjust row references per department row).

For summary statistics in H42:L47 (six rows), write:
- MIN over the 6 department cells in that column (e.g., `=MIN(H35:H40)`)
- MAX (e.g., `=MAX(H35:H40)`)
- MEDIAN (e.g., `=MEDIAN(H35:H40)`)
- AVERAGE (e.g., `=AVERAGE(H35:H40)`)
- 25th percentile: use `=PERCENTILE(H35:H40,0.25)` — but **prefix it as `=_xlfn.PERCENTILE.INC(H35:H40,0.25)`** when writing via openpyxl so it doesn't produce #NAME? in the verifier. Alternatively, use the classic `=PERCENTILE(H$35:H$40,0.25)` form. **Important**: After writing, re-read the cell to confirm. If the library writes `PERCENTILE` without issues, keep it. If the verifier environment needs `_xlfn.` prefix, use it.
- 75th percentile: same pattern with 0.75.

**Critical note from prior failure**: The previous attempt produced #NAME? for percentile rows. The safest approach:
  - Try `=PERCENTILE(H35:H40,0.25)` first (the classic Excel function name).
  - If openpyxl or the verifier rejects it, use `=_xlfn.PERCENTILE.INC(H35:H40,0.25)`.
  - Verify by re-reading the cell value after writing.

Match the row assignment (min/max/median/mean/25th/75th) to the actual labels found in Phase 0.

## Phase 3 — Weighted mean (H50:L50)
For each column (H through L):
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

## Phase 4 — Save and verify
1. `mkdir -p /root/output`
2. Save as `/root/output/result.xlsx`.
3. Re-open the saved file and print:
   - A sample lookup cell (e.g., H12) to confirm it has a formula.
   - A sample Net budget buffer cell (e.g., H35).
   - Cells H46 and H47 to confirm they do NOT contain #NAME? — print the raw formula string.
   - Cell H50 for the weighted mean formula.
4. If any cell shows #NAME? or unexpected content, fix it before finishing.

## Constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Work only inside sheets Task and Data (and only write to Task).
- Do not use data_only=True when writing; preserve formulas.

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