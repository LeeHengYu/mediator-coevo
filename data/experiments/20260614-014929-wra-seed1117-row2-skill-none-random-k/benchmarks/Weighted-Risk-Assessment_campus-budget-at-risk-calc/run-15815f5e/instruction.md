# Task Instruction

## Task: Populate formulas and calculations in /root/data/workbook.xlsx

### Phase 0 — Investigation (Do NOT edit any files yet)

1. Copy the workbook for safety: `cp /root/data/workbook.xlsx /root/data/workbook_backup.xlsx`
2. Create and run a Python script to inspect the workbook structure:
   - List all sheet names
   - On sheet `Task`: print all cell values in rows 10-50 for columns A-L. Pay special attention to:
     - Row 10 (years)
     - Column D rows 12-17, 19-24, 26-31 (series codes)
     - What labels are in column A/B/C for rows 12-17, 19-24, 26-31 (the three lookup blocks)
     - What labels are in rows 35-40 (Net budget buffer departments), rows 42-47 (statistics labels), row 50 (weighted mean)
     - Check which cells in H12:L17, H19:L24, H26:L31 are empty (these are the yellow cells to fill)
   - On sheet `Data`: print all cell values in rows 21-38 for ALL columns to understand the lookup table structure. Identify:
     - Which column contains series codes
     - Which row contains years
     - The data layout (is it vertical with series codes in a column, or horizontal?)
3. Print the findings clearly before proceeding.

### Phase 1 — Lookup Formulas (H12:L17, H19:L24, H26:L31)

Based on Phase 0 findings, populate the yellow cells with INDEX/MATCH formulas. Each formula should:
- Use the series code from column D of the current row
- Use the year from row 10 of the current column
- Look up values from the `Data` sheet rows 21:38
- Use INDEX with two MATCH functions (one for row, one for column) — this is the most reliable pattern
- Use appropriate absolute/relative references so formulas can be placed in each cell correctly
- Example pattern: `=INDEX(Data!$B$22:$Z$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$Z$21, 0))`
  - **Adjust the actual ranges based on Phase 0 findings** — the column where series codes live, the row where years live, and the data range boundaries.

IMPORTANT: When writing formulas with openpyxl, set `cell.value = '=FORMULA...'` as a string starting with `=`. Use the `Task` sheet reference implicitly (no sheet prefix for same-sheet refs like `$D12` or `H$10`) and prefix cross-sheet refs with `Data!`.

### Phase 2 — Net Budget Buffer (H35:L40)

Identify which of the three blocks (rows 12-17, 19-24, 26-31) corresponds to:
- Committed Funding
- Operating Spend  
- Approved Budget Base

Then for each cell in H35:L40, write the formula:
`=(CommittedFunding - OperatingSpend) / ApprovedBudgetBase * 100`

For example, if block 1 (rows 12-17) is Committed Funding, block 2 (rows 19-24) is Operating Spend, and block 3 (rows 26-31) is Approved Budget Base, then H35 would be:
`=(H12-H19)/H26*100`

Adjust row references to align each department correctly across blocks.

### Phase 3 — Statistics (H42:L47)

For each column H through L, calculate these statistics over the 6 Net budget buffer values (e.g., H35:H40):
- Row 42: MIN → `=MIN(H35:H40)` (or `=MIN(H$35:H$40)`)
- Row 43: MAX → `=MAX(H35:H40)`
- Row 44: MEDIAN → `=MEDIAN(H35:H40)`
- Row 45: AVERAGE → `=AVERAGE(H35:H40)`
- Row 46: 25th percentile → `=PERCENTILE(H35:H40,0.25)`
- Row 47: 75th percentile → `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC` — openpyxl and older Excel compatibility requires the basic `PERCENTILE` function name. The failed run artifact shows #NAME? errors from using unsupported function names.

Verify the order of statistics by checking the labels in column A/B/C/D for rows 42-47.

### Phase 4 — Weighted Mean (H50:L50)

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of Net budget buffer percentages weighted by Approved Budget Base.

Alternatively, if the instruction says "use SUMPRODUCT", the formula is:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

### Phase 5 — Save and Validate

1. Save the workbook to `/root/output/result.xlsx` (create `/root/output/` directory if needed).
2. Reopen the saved file and verify:
   - All cells in H12:L17, H19:L24, H26:L31 contain formula strings (start with `=`)
   - All cells in H35:L40 contain formula strings
   - All cells in H42:L47 contain formula strings
   - All cells in H50:L50 contain formula strings
   - No cells contain `#NAME?`, `None`, or empty values where formulas should be
   - The workbook still has exactly the original sheets (Task and Data, no extras)
3. Print a summary of all formulas written.

### Key Constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs
- Do NOT change existing formatting
- Use `openpyxl` to read and write the workbook
- When loading, do NOT use `data_only=True` (we need to preserve and write formulas)
- Ensure all cross-sheet references use the exact sheet name from the workbook (likely `Data`)

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