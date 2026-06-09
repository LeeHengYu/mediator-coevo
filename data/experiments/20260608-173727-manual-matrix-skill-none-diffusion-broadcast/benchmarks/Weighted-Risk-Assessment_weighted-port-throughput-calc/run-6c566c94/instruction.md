# Task Instruction

Execute the following steps carefully to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and inspect:
- Sheet names (confirm `Task` and `Data` exist)
- On sheet `Task`: read row 10 (especially H10:L10) to see the years; read column D rows 12-17, 19-24, 26-31 to see the series codes; read rows 35-40 column D to see port names; read H35:L40, H42:L47, H50:L50 to confirm they are empty/yellow target cells; read any labels in rows 42-47 column D-G to see what stats are expected (min, max, median, mean, 25th, 75th percentile).
- On sheet `Data`: read rows 21-38 to understand the data layout — identify which row is the header row, which column has series codes, and how years are arranged (likely years in columns). Print out rows 20-38 fully (all columns with data) so you understand the exact structure.
- Print cell values and any existing formulas in the target ranges to confirm they're empty.

## 2. Determine the lookup structure
Based on the Data sheet inspection:
- Identify the column that contains series codes (likely column A or B on Data sheet).
- Identify the row that contains years (likely row 20 or 21 on Data sheet).
- Determine the exact range for the lookup table (rows 21:38, but note which columns).
- Choose a lookup pattern. A good default is INDEX-MATCH:
  `=INDEX(Data!$B$21:$Z$38, MATCH(D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
  But you MUST adjust the exact column letters and row numbers based on what you actually see in the Data sheet. The series code column and year header row must be identified from inspection.

## 3. Populate lookup formulas (Step 1)
Using openpyxl, write formulas into the yellow cells. For each cell in ranges H12:L17, H19:L24, H26:L31:
- The formula should reference the series code from column D of that row (e.g., `$D12` for row 12) and the year from row 10 of that column (e.g., `H$10` for column H).
- Use INDEX-MATCH or another approved pattern. Make sure references use the correct absolute/relative anchoring so formulas can be placed in each cell correctly.
- Use the actual Data sheet range you discovered. For example, if Data has series codes in column A rows 21-38 and year headers in row 20 starting from column B:
  `=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))`
  Replace $XX with the actual last data column.

IMPORTANT: When writing formulas with openpyxl, the cell value must be a string starting with '='. Do NOT use data_only mode for writing.

## 4. Net container flow formulas (Step 2 first part)
For H35:L40, write formulas that calculate:
`(Loaded Containers Inbound - Loaded Containers Outbound) / Terminal Throughput Capacity * 100`

These correspond to the three blocks above:
- H12:L17 = one metric (e.g., Loaded Containers Inbound)
- H19:L24 = another metric (e.g., Loaded Containers Outbound)  
- H26:L31 = Terminal Throughput Capacity

Verify which block is which by checking the labels in column D or nearby. The six ports in rows 35-40 should correspond to the six ports in rows 12-17 (and 19-24, 26-31).

For each cell, e.g., H35: `=(H12-H19)/H26*100` (adjust row references based on actual port ordering — make sure port in row 35 matches port in rows 12, 19, 26).

## 5. Summary statistics (Step 2 second part)
For H42:L47, write column-wise formulas over H35:L40. Based on the labels in column D for rows 42-47, assign:
- Minimum: `=MIN(H35:H40)`
- Maximum: `=MAX(H35:H40)`
- Median: `=MEDIAN(H35:H40)`
- Simple mean: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Match the exact row to the exact label. Read the labels carefully.

## 6. Weighted mean (Step 3)
For H50:L50, write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of Net container flow percentages using Terminal Throughput Capacity as weights.

## 7. Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change any formatting, do not add sheets, macros, or VBA.

## 8. Verify
Reopen the saved file and confirm:
- All target cells contain formulas (not empty, not plain values)
- Formula strings look correct
- No extra sheets were added
- Print a sample of formulas from each target range

## Critical Notes
- You MUST inspect the actual workbook structure before writing any formulas. Do not assume column letters or row numbers.
- Adjust all cell references based on what you actually observe.
- When writing formulas with openpyxl, just assign a string like `cell.value = '=INDEX(...)'`.
- Preserve all existing formatting — load with openpyxl keeping styles (do not use data_only=True for the workbook you save).
- If the Data sheet has merged cells or unusual structure, adapt accordingly.
- Double-check that the port order in rows 35-40 matches the port order in rows 12-17, 19-24, 26-31. If they differ, cross-reference by port name in column D.

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