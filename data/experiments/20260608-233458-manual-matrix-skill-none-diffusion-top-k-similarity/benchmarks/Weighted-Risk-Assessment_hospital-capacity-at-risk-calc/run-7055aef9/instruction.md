# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Phase 0 – Inspect the workbook layout
1. Load /root/data/workbook.xlsx with openpyxl (do NOT use data_only=True).
2. Print the sheet names to confirm 'Task' and 'Data' exist.
3. On sheet 'Task':
   - Print rows 10-50, columns A-L (values and any existing formulas) so you can see:
     • The year headers in row 10 (columns H-L).
     • The series codes in column D for rows 12-17, 19-24, 26-31.
     • The labels/structure of rows 35-40 (Net capacity headroom), 42-47 (stats), and row 50 (weighted mean).
4. On sheet 'Data':
   - Print rows 20-40, all populated columns, to see the data orientation, column headers, series codes, and year labels. Identify exactly which row holds years and which column holds series codes. Note the exact cell range boundaries (e.g., Data!$B$21:$B$38 for series codes, Data!$C$21:$G$21 for years, Data!$C$22:$G$38 for values — adjust after inspection).

Do NOT proceed to formula writing until you have printed and understood both sheets.

## Phase 1 – Populate lookup formulas in H12:L31
For every cell in the three blocks H12:L17, H19:L24, H26:L31, write an INDEX-MATCH-MATCH formula that:
- Uses the series code from column D of the same row (e.g., $D12 for row 12).
- Uses the year from row 10 of the same column (e.g., H$10 for column H).
- Looks up in the Data sheet range you identified in Phase 0.
- Pattern: =INDEX(Data!<value_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
- Use absolute references for the Data ranges and mixed references ($D12, H$10) so the formula copies correctly across the block.

Write these formulas programmatically in a loop over the rows and columns.

## Phase 2 – Net capacity headroom (H35:L40)
For each of the six hospital clusters (rows 35-40), write a formula:
  =(H12 - H19) / H26 * 100
adjusted so that:
- Row 35 uses the Available Care Slots from row 12, Occupied Care Slots from row 19, Staffed Bed Capacity from row 26.
- Row 36 uses rows 13, 20, 27. And so on for all six clusters.
- Use relative references so the pattern is consistent across columns H-L.

Verify by inspecting the Task sheet that rows 12-17 = Available Care Slots, 19-24 = Occupied Care Slots, 26-31 = Staffed Bed Capacity. If the mapping differs, adjust accordingly based on your Phase 0 inspection.

## Phase 3 – Summary statistics (H42:L47)
For each column H through L, write these formulas in the corresponding rows:
- Row 42: =MIN(H35:H40)
- Row 43: =MAX(H35:H40)
- Row 44: =MEDIAN(H35:H40)
- Row 45: =AVERAGE(H35:H40)
- Row 46: =PERCENTILE(H35:H40, 0.25)
- Row 47: =PERCENTILE(H35:H40, 0.75)

Verify the row-label mapping from Phase 0 (minimum, maximum, median, mean, 25th percentile, 75th percentile). Adjust row assignments if the labels differ.

## Phase 4 – Weighted mean (H50:L50)
For each column H through L:
  =SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
This computes the weighted mean of the Net capacity headroom percentages using Staffed Bed Capacity as weights.

## Phase 5 – Save and verify
1. Create /root/output/ directory if it doesn't exist.
2. Save the workbook to /root/output/result.xlsx.
3. Reload the saved file and print a sample of cells (e.g., H12, H35, H42, H50) to confirm formulas are present (not None or bare values).
4. Confirm no new sheets were added and the file is saved correctly.

## Important constraints
- Do NOT use data_only=True when loading.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting.
- Inspect before writing. Adjust all cell references based on actual workbook layout discovered in Phase 0.

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