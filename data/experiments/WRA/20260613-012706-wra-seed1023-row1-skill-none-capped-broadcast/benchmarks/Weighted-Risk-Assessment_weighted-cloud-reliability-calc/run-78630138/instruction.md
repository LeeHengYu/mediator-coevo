# Task Instruction

You must update the workbook `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Use Python with openpyxl. Do NOT use xlsxwriter. Do NOT evaluate formulas—just write Excel formula strings into cells. Follow these steps precisely:

## Preliminary
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl, keeping existing formatting (`load_workbook` with default args—do NOT pass `data_only=True`).
3. Inspect sheets `Task` and `Data` to understand the layout:
   - On `Task`: read column D for series codes in rows 12-17, 19-24, 26-31. Read row 10 columns H-L for years.
   - On `Data`: inspect rows 21-38 to understand the data table structure (which row has headers, which column has series codes, where years appear).
   - Print out what you find so you can build correct formulas.

## Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write an Excel formula that looks up the value from sheet `Data` rows 21:38. The formula must use the series code from column D of that row on `Task` and the year from row 10 of `Task`.

Use INDEX/MATCH/MATCH pattern. For example (adjust ranges/sheets after inspecting the actual data layout):
```
=INDEX(Data!$B$22:$XX$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$XX$21, 0))
```
Adjust the actual column/row ranges based on what you observe in the Data sheet. The key point: the row lookup matches the series code in column D against the series code column in Data, and the column lookup matches the year in row 10 against the year header row in Data.

IMPORTANT: Write formulas as strings (e.g., `ws['H12'] = '=INDEX(...)'`). Do NOT try to compute values in Python.

## Step 2: Net reliability gap (H35:L40) and statistics (H42:L47)
For H35:L40, write formulas computing:
`(Successful API Requests - Failed API Requests) / Compute Capacity * 100`

The three blocks are:
- Successful API Requests: rows 12-17
- Failed API Requests: rows 19-24  
- Compute Capacity: rows 26-31

So for cell H35: `=(H12-H19)/H26*100`, for H36: `=(H13-H20)/H27*100`, etc. through all 6 regions and 5 year columns.

For H42:L47, write column-wise statistics over H35:L40:
- Row 42: MIN  → `=MIN(H35:H40)`
- Row 43: MAX  → `=MAX(H35:H40)`
- Row 44: MEDIAN → `=MEDIAN(H35:H40)`
- Row 45: AVERAGE (simple mean) → `=AVERAGE(H35:H40)`
- Row 46: 25th percentile → `=PERCENTILE(H35:H40,0.25)`
- Row 47: 75th percentile → `=PERCENTILE(H35:H40,0.75)`

CRITICAL: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid #NAME? errors in older Excel engines. Similarly use `MEDIAN`, `MIN`, `MAX`, `AVERAGE` — all standard functions.

However, FIRST check what labels exist in the Task sheet for rows 42-47 to confirm the correct order of statistics. Adjust the row assignments if the labels differ from the order above.

## Step 3: Weighted mean in H50:L50
For each column H through L, write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the Net reliability gap percentages as values and Compute Capacity as weights.

## Final
- Save to `/root/output/result.xlsx`.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify any existing formatting.
- After saving, reopen the file and print a few formula cells to confirm they were written correctly as formula strings (start with '=').
- Verify no cells contain Python-computed floats where formulas should be.

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