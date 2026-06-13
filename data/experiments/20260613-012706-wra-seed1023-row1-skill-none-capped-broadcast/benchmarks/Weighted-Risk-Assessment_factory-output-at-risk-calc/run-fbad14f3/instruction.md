# Task Instruction

You must update the Excel workbook at `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Follow these steps precisely:

## Step 0: Inspect the workbook structure
1. `mkdir -p /root/output`
2. Use Python with openpyxl to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: read row 10 (the year headers in columns H–L), column D rows 12–17, 19–24, 26–31 to get the series codes. Read any labels in column A or B for those row groups to understand what data blocks they represent (e.g., Finished Output, Scrap And Rework, Rated Production Capacity). Also read rows 35–40 column D for series codes or labels, rows 42–47 column A–G for stat labels, and row 50 for the weighted mean label.
   - Sheet `Data`: read rows 21–38 to understand the data layout — which row/column holds the series code, and where the year-indexed values are.
3. Print all of this so you can build correct formulas.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in those ranges, write an Excel formula that retrieves data from sheet `Data` rows 21:38. Use the `INDEX`/`MATCH` pattern:
- The lookup key is the series code in column D of the current row on `Task`.
- The column selection is based on matching the year in row 10 of `Task` against the header row in `Data`.
- Make sure to identify exactly which row in `Data` contains the headers (years) and which column contains the series codes, then build the INDEX/MATCH accordingly.
- Use appropriate absolute/mixed references so formulas are correct across the range.

## Step 2: Net production slack in H35:L40 and statistics in H42:L47
For H35:L40, compute: `(Finished_Output - Scrap_And_Rework) / Rated_Production_Capacity * 100` using cell references to the blocks populated in Step 1. Identify which block (rows 12–17, 19–24, or 26–31) corresponds to Finished Output, Scrap And Rework, and Rated Production Capacity respectively.

For H42:L47, compute column-wise statistics over H35:L40:
- Row 42: MIN
- Row 43: MAX  
- Row 44: MEDIAN
- Row 45: AVERAGE
- Row 46: PERCENTILE (25th) — use `PERCENTILE(range, 0.25)` or `PERCENTILE.INC(range, 0.25)`
- Row 47: PERCENTILE (75th) — use `PERCENTILE(range, 0.75)` or `PERCENTILE.INC(range, 0.75)`

**CRITICAL**: Check the labels in column A/B/C for rows 42–47 to determine the correct order of MIN, MAX, MEDIAN, MEAN, 25th, 75th percentile. Do NOT assume the order above — use whatever order the labels specify.

**CRITICAL**: Use `PERCENTILE.INC` (not `PERCENTILE.EXC`) for the percentile formulas. The function `PERCENTILE` also works. Do NOT use any function name that might cause #NAME? errors — verify the function name is valid in Excel. Avoid `PERCENTILE.EXC` unless the labels specifically say "exclusive".

## Step 3: Weighted mean in H50:L50
For each column H–L, use: `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)` (adjusting column letters). This computes the weighted mean of the slack percentages weighted by Rated Production Capacity.

## Step 4: Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.

## Step 5: Validate
Reopen `/root/output/result.xlsx` and print the formulas in a few sample cells (e.g., H12, L17, H35, H42, H46, H47, H50) to confirm they are syntactically correct Excel formulas. Also check that no cells contain plain values where formulas are expected.

If any test exists at `/root/tests/`, run it with `cd /root && python -m pytest tests/ -v` and report results.

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