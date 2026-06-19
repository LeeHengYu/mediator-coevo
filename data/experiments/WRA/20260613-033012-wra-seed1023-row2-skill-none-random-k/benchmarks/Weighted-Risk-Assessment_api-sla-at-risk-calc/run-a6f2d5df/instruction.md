# Task Instruction

Execute the following steps exactly to produce `/root/output/result.xlsx`.

## 0. Preparation
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Inspect:
- Sheet `Task`: read the series codes in column D for rows 12-17, 19-24, 26-31 (these are the six services repeated in three blocks). Read the years in row 10 for columns H-L. Read what labels are in column D or G for rows 35-40, 42-47, 50. Note any existing content/formatting.
- Sheet `Data`: read rows 21-38 to understand the layout (column headers, row labels, data orientation). Determine whether the data is arranged so that series codes are in a column (for VLOOKUP) or a row (for HLOOKUP), and where years appear.

Print all of this information before writing any formulas.

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write an Excel formula that combines the series code from column D of that row with the year from row 10 of that column to look up the value from `Data!$21:$38`.

Use INDEX/MATCH pattern. Example (adjust references after inspection):
- If Data has series codes in column A rows 21-38 and years in row 20 columns B onward:
  `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
- Adapt column/row references to match the actual layout you discover in step 1.
- Use absolute references for the Data range and mixed references ($D12 for row-lock on column, H$10 for column-lock on row) so formulas copy correctly across the block.

## 3. Net SLA buffer in H35:L40
For each of the six services (rows 35-40), compute:
`= (corresponding_Latency_Budget_Preserved - corresponding_Latency_Budget_Consumed) / corresponding_Covered_Request_Capacity * 100`

The three blocks are:
- H12:L17 = first metric block (check which one is Latency Budget Preserved, Latency Budget Consumed, or Covered Request Capacity by reading the block header labels near rows 11, 18, 25)
- H19:L24 = second metric block
- H26:L31 = third metric block

Map the formula terms to the correct blocks based on the actual labels.

## 4. Statistics in H42:L47
For each column H through L, compute column-wise statistics over the six Net SLA buffer values (H35:H40 etc.):
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)` — **Use AVERAGE, NOT MEAN. Excel has no MEAN function.**
- Row 46: `=PERCENTILE.INC(H35:H40, 0.25)` — **Use PERCENTILE.INC (or PERCENTILE), NOT PERCENTILE.EXC**
- Row 47: `=PERCENTILE.INC(H35:H40, 0.75)`

IMPORTANT: Check which row corresponds to which statistic by reading the labels in column D/G for rows 42-47. The previous failure was caused by using 'MEAN' instead of 'AVERAGE' and possibly wrong PERCENTILE syntax. Assign formulas to the correct rows based on actual labels. The mapping above (42=min, 43=max, etc.) is a guess — verify from the sheet labels.

## 5. Weighted mean in H50:L50
For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of Net SLA buffer percentages weighted by Covered Request Capacity. Adjust the Covered Request Capacity range (H26:H31) if inspection shows it's in a different block.

## 6. Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change any existing formatting, do NOT add sheets, macros, VBA, or external links.

## 7. Validate
Reopen the saved file with openpyxl (data_only=False) and print the formula strings in cells H42:L47 and H50:L50 to confirm:
- No cell contains 'MEAN' — should be 'AVERAGE'
- PERCENTILE.INC is used (not an undefined name)
- All formula cells are non-empty

Also spot-check a few lookup cells (e.g., H12, L31) to confirm formulas are present.

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