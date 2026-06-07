# Task Instruction

Complete the following task to update an Excel workbook with formulas and calculations.

## Setup
1. First, copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Inspect the workbook structure thoroughly before making any changes:
   - Read sheet `Task`: examine the layout of columns D (series codes), row 10 (years), and the yellow cell ranges H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50.
   - Read sheet `Data`: examine rows 21:38 to understand the data layout (column headers, row labels, how series codes and years map to values).
   - Note the exact series codes in column D for rows 12-17, 19-24, 26-31, and 35-40.
   - Note the exact years in row 10 for columns H through L.
   - Note the exact structure of the Data sheet rows 21:38 — which column contains series codes, which row contains years, and where the data values are.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a spreadsheet formula that looks up a value from `Data!$21:$38` using:
- The series code from column D of the current row (on sheet Task)
- The year from row 10 of the current column (on sheet Task)

Use one of these patterns: `INDEX(MATCH, MATCH)`, `VLOOKUP(MATCH)`, `HLOOKUP(MATCH)`, or `XLOOKUP(MATCH)`. Choose the pattern that best fits the Data sheet layout.

IMPORTANT: Before writing formulas, carefully determine:
- Whether the Data sheet has series codes in a column or row
- Whether years are in a row or column
- The exact range references needed
- Use appropriate absolute references ($) so formulas can be filled across the range correctly (lock the lookup column/row references but allow the series code row and year column to vary)

## Step 2: Net reliability gap and statistics in H35:L47

For H35:L40 (6 regions), calculate:
`Net reliability gap = (Successful API Requests - Failed API Requests) / Compute Capacity * 100`

- "Successful API Requests" values should be in H12:L17
- "Failed API Requests" values should be in H19:L24  
- "Compute Capacity" values should be in H26:L31
- Verify this mapping by checking the actual labels on the Task sheet before writing formulas.

The formula for each cell in H35:L40 should reference the corresponding cells from the three blocks above. For example, H35 = (H12 - H19) / H26 * 100 (adjust if the row mapping differs).

For H42:L47, calculate column-wise statistics over H35:L40:
- Row 42: MIN (minimum)
- Row 43: MAX (maximum)
- Row 44: MEDIAN
- Row 45: AVERAGE (simple mean)
- Row 46: PERCENTILE (25th percentile, i.e., PERCENTILE(H35:H40, 0.25))
- Row 47: PERCENTILE (75th percentile, i.e., PERCENTILE(H35:H40, 0.75))

IMPORTANT: Check the actual labels in column D or nearby columns for rows 42-47 to confirm which statistic goes in which row. Adjust the row assignments above if the labels differ.

## Step 3: Weighted mean in H50:L50

For each column H through L, calculate:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

Wait — the instruction says use SUMPRODUCT with Step 2 percentages as values and Compute Capacity as weights. The weighted mean formula should be:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted average of the net reliability gap percentages weighted by compute capacity.

## Constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Write all formulas as Excel formulas (not hardcoded values).
- Save the final result to `/root/output/result.xlsx`.

## Validation
- After writing all formulas, open the result file and verify that the formula cells contain formulas (not raw values).
- Spot-check a few lookup formulas to confirm they return numeric values (not errors).
- Verify the statistics formulas reference the correct ranges.

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