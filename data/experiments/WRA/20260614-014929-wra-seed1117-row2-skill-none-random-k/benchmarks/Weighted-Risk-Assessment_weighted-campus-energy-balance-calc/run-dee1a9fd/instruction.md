# Task Instruction

Implement the weighted campus energy balance workbook task. Follow these steps precisely:

## Setup
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Open `/root/output/result.xlsx` with openpyxl (keep formulas, not cached values).

## Inspection (Critical)
3. Inspect the `Data` sheet:
   - Read row 20 to find the year headers and their column positions.
   - Read column A rows 21–38 to find the series codes and their row positions.
   - Print all of these so you have the exact mapping.
4. Inspect the `Task` sheet:
   - Read column D rows 12–17, 19–24, 26–31 to get the series codes used in lookups.
   - Read row 10 columns H–L to get the year headers.
   - Read column B/C rows 12–17, 19–24, 26–31 to understand the three blocks (Renewable Generation, Grid Consumption, Baseline Energy Demand).
   - Read rows 35–40 column B/C/D to see campus labels for Net renewable balance.
   - Read rows 42–47 column B/C/D/E/F/G to see the statistical function labels (Min, Max, Median, Mean, 25th percentile, 75th percentile) — verify their exact order.
   - Read row 50 to see the MCEC weighted mean label.
   - Print everything so the exact layout is confirmed before writing any formulas.

## Step 1: Lookup Formulas in H12:L31
5. For each cell in the three blocks H12:L17, H19:L24, H26:L31, write an INDEX/MATCH formula that:
   - Uses the series code from column D of that row (e.g., `$D12`)
   - Uses the year from row 10 of that column (e.g., `H$10`)
   - Looks up from Data sheet rows 21:38, with codes in Data column A and years in Data row 20.
   - Use absolute references for the Data sheet ranges so formulas copy correctly.
   - Example pattern: `=INDEX(Data!$B$21:$XX$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$XX$20,0))`
   - Adjust the column range endpoints based on what you found in step 3 (use the actual last data column).

## Step 2: Net Renewable Balance (H35:L40)
6. Determine which rows in the Task sheet correspond to each campus for Renewable Generation (rows 12–17), Grid Consumption (rows 19–24), and Baseline Energy Demand (rows 26–31). The six campuses should be in the same order.
7. For each cell in H35:L40, write the formula:
   `=(H12-H19)/H26*100` (adjusting row references for each campus row offset).
   - Row 35 uses rows 12, 19, 26
   - Row 36 uses rows 13, 20, 27
   - Row 37 uses rows 14, 21, 28
   - Row 38 uses rows 15, 22, 29
   - Row 39 uses rows 16, 23, 30
   - Row 40 uses rows 17, 24, 31

## Step 2 continued: Statistics (H42:L47)
8. Based on the verified label order from step 4, assign the correct Excel function to each row. Map each label to its function:
   - Min → `=MIN(H35:H40)`
   - Max → `=MAX(H35:H40)`
   - Median → `=MEDIAN(H35:H40)`
   - Mean → `=AVERAGE(H35:H40)`
   - 25th percentile → `=PERCENTILE(H35:H40,0.25)`
   - 75th percentile → `=PERCENTILE(H35:H40,0.75)`
   Write formulas in H42:L47 accordingly, adjusting column for each.

## Step 3: Weighted Mean (H50:L50)
9. For each cell in H50:L50, write:
   `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`
   Adjusting the column letter for each.

## Validation
10. Re-read a sample of cells (e.g., H12, L17, H35, L40, H42, H47, H50, L50) and confirm each `.value` is a string starting with `=`.
11. Save the workbook.
12. Print 'DONE' when complete.

## Constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change any existing formatting.
- Use openpyxl only. Do not use xlsxwriter or other libraries for writing.

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