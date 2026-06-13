# Task Instruction

Execute the following steps in a single Python script using openpyxl.

## Step 0 – Inspect the workbook
Open `/root/data/workbook.xlsx` with `data_only=False`.
Print the following so you can see the exact layout before writing any formulas:
- Task sheet: rows 10-50, columns A-L (print cell coordinates + values)
- Data sheet: rows 1-40, columns A-Z (print cell coordinates + values)

Pay special attention to:
- Row 10 on Task: which columns (H-L) hold the year headers
- Column D on Task: which rows hold series codes for blocks H12:L17, H19:L24, H26:L31
- Data sheet: the exact row range and column layout (where series codes live, where years start)

## Step 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31
Using the layout discovered in Step 0, write INDEX/MATCH formulas into every cell in the three blocks. Each formula should:
- Use INDEX over the data value area on the Data sheet (rows 21-38 or wherever the data actually lives)
- Use MATCH on the series code from column D of the current Task row against the series-code column on Data
- Use MATCH on the year from row 10 of the current Task column against the year header row on Data
- Use absolute references for the Data ranges and mixed references so the formula is correct in every cell

Do NOT hardcode values. Write actual Excel formulas as strings.

## Step 2 – Net capacity headroom (H35:L40)
For each of the 6 hospital clusters (rows 35-40) and each year column (H-L), write a formula:
`= (AvailableCareSlots - OccupiedCareSlots) / StaffedBedCapacity * 100`
where:
- AvailableCareSlots comes from the first lookup block (H12:L17)
- OccupiedCareSlots comes from the second lookup block (H19:L24)
- StaffedBedCapacity comes from the third lookup block (H26:L31)

The row offset within each block should match: row 35 uses row 12, 19, 26; row 36 uses row 13, 20, 27; etc.

## Step 3 – Summary statistics (H42:L47)
For each year column (H-L), write formulas for:
- Row 42: MIN of H35:H40 (same column)
- Row 43: MAX of H35:H40
- Row 44: MEDIAN of H35:H40
- Row 45: AVERAGE of H35:H40
- Row 46: PERCENTILE(H35:H40, 0.25)
- Row 47: PERCENTILE(H35:H40, 0.75)

## Step 4 – Weighted mean (H50:L50)
For each year column (H-L), write a SUMPRODUCT formula:
`= SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
using the Net capacity headroom percentages as values and the Staffed Bed Capacity block as weights.

## Step 5 – Save
Save the workbook to `/root/output/result.xlsx`. Create the output directory if needed. Do NOT change any existing formatting, do NOT add sheets.

## Important execution notes
- Run the inspection (Step 0) output FIRST. Read it carefully. If the actual layout differs from the row/column assumptions above (e.g., data rows are not 21-38, or year headers are on a different row), adjust all subsequent formulas accordingly before writing them.
- After writing all formulas, re-read a sample of cells (e.g., H12, H35, H42, H50) and print their values to confirm formulas were written.
- Use a two-pass approach: first code block inspects and prints; second code block writes formulas based on what you learned.

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