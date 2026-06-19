# Task Instruction

Execute the following steps in order to produce /root/output/result.xlsx.

## Step 0 – Inspect the workbook

Open /root/data/workbook.xlsx with openpyxl (data_only=False). Print:
1. Sheet names.
2. From sheet 'Task': cells A10:L10 (year headers), cells A12:G17 (series codes block 1), A19:G24 (block 2), A26:G31 (block 3), A35:G40 (net renewable balance area), A42:G47 (stats area), A50:G50 (weighted mean row).
3. From sheet 'Data': cells A1:Z1 (header row if any), cells A21:Z38 (the lookup data block). Print enough to see the structure: row indices, column letters, and actual cell values.

Do NOT proceed to Step 1 until you have printed and read this output. All subsequent formulas depend on the exact layout discovered here.

## Step 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31

For every yellow cell in these three blocks, write an Excel formula that combines INDEX and MATCH (or an equivalent approved pattern) to look up the value from the Data sheet rows 21:38. Each formula must use two inputs:
- The series code in column D of the current row on 'Task'.
- The year in row 10 of the current column on 'Task'.

Adjust the exact Data-sheet ranges ($-anchoring, row span, column span) based on what you discovered in Step 0. Use absolute references for the data range and mixed references so the formula can be placed in each cell correctly (or write each cell's formula individually if simpler).

After writing, re-read a sample cell (e.g., H12, L17, H26) to confirm the formula string is stored (not None).

## Step 2 – Net renewable balance (H35:L40) and summary statistics (H42:L47)

Net renewable balance formula per cell:
  = (CellFromBlock1 - CellFromBlock2) / CellFromBlock3 * 100

where Block1 = Renewable Generation (rows 12:17), Block2 = Grid Consumption (rows 19:24), Block3 = Baseline Energy Demand (rows 26:31). Map each campus row (35→12/19/26, 36→13/20/27, … 40→17/24/31) and each year column (H–L).

Summary statistics in rows 42–47, for each column H–L:
- Row 42: =MIN(H35:H40)
- Row 43: =MAX(H35:H40)
- Row 44: =MEDIAN(H35:H40)
- Row 45: =AVERAGE(H35:H40)
- Row 46: =PERCENTILE(H35:H40,0.25)
- Row 47: =PERCENTILE(H35:H40,0.75)

Verify the row labels in column A/B/C/D match the expected statistic names; adjust row assignments if the actual layout differs.

## Step 3 – Weighted mean in H50:L50

For each column c in H–L:
  =SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
(Replace H with the appropriate column letter for each cell.)

This uses the net renewable balance percentages as values and the Baseline Energy Demand block as weights.

## Step 4 – Save and verify

1. Save the workbook to /root/output/result.xlsx (create /root/output/ if needed).
2. Re-open the saved file with openpyxl (data_only=False).
3. Print the formula strings in cells H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50.
4. Confirm none of them are None.

Do not add any new sheets, macros, VBA, external links, or helper tabs. Do not alter existing formatting.

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