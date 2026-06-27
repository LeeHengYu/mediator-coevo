# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook
Open /root/data/workbook.xlsx with openpyxl (data_only=False). Print:
- Sheet names
- Task!D12:D17 (series codes for block 1), Task!D19:D24 (block 2), Task!D26:D31 (block 3)
- Task!H10:L10 (years row)
- Data!A21:A38 (series codes in Data sheet)
- Data!C20:G20 (years header row in Data sheet)
- Data!C21:G38 (the actual data values)
- Task!H35:H40, Task!D35:D40 (region labels or row references for Net reliability gap)
- Task!H42:H47 row labels (min, max, median, mean, 25th, 75th)
- Any existing content in H50:L50

This inspection is critical: note the exact column letters and row numbers on the Data sheet where years appear and where data values live. Also check whether the Data sheet uses column A for series codes or some other column, and whether years are in a row header.

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write an INDEX-MATCH formula that:
- Uses the series code from column D of the same row on sheet Task
- Uses the year from row 10 of the same column on sheet Task
- Looks up the value from the Data sheet rows 21:38

Based on the inspection, construct the formula. The typical pattern will be:
```
=INDEX(Data!$C$21:$G$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$C$20:$G$20, 0))
```
Adjust the column letters and row numbers based on what you actually find in the inspection step. The $ signs must anchor the lookup ranges but allow the series code reference ($D12) to change rows and the year reference (H$10) to change columns.

Write these formulas using openpyxl by assigning the formula string to each cell. Do NOT use data_only mode when writing.

## 3. Net reliability gap in H35:L40
From the inspection, identify which rows in the Task sheet correspond to:
- Successful API Requests (one of the blocks H12:L17)
- Failed API Requests (another block H19:L24)
- Compute Capacity (the block H26:L31)

The formula for each cell in H35:L40 is:
```
=(H12-H19)/H26*100
```
(Adjust row references based on which block maps to which metric. The six regions should correspond row-by-row: row 35 uses data from rows 12, 19, 26; row 36 uses 13, 20, 27; etc.)

Write these as Excel formulas.

## 4. Statistics in H42:L47
For each column H through L, write:
- Row 42: =MIN(H35:H40)
- Row 43: =MAX(H35:H40)
- Row 44: =MEDIAN(H35:H40)
- Row 45: =AVERAGE(H35:H40)
- Row 46: =PERCENTILE(H35:H40,0.25)  — use PERCENTILE.INC or PERCENTILE depending on what works. **IMPORTANT**: Use `PERCENTILE.INC` as the function name since this is the standard Excel function. If the avoid-artifact warns about #NAME? errors, test both `PERCENTILE` and `PERCENTILE.INC`. In openpyxl, write `PERCENTILE.INC` (with the dot) — Excel recognizes this. Actually, the safest choice is plain `PERCENTILE` which is universally recognized.
- Row 47: =PERCENTILE(H35:H40,0.75)  — same function choice as row 46.

**Critical**: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid #NAME? errors. The plain `PERCENTILE` function is recognized by all Excel engines including the verifier's evaluation engine.

## 5. Weighted mean in H50:L50
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net reliability gap percentages using Compute Capacity as weights.

## 6. Save
Save the workbook to /root/output/result.xlsx. Do NOT change any formatting, do not add sheets, macros, or named ranges.

## 7. Verify
Reopen /root/output/result.xlsx and print:
- A sample of the lookup formulas (e.g., H12, L17)
- A sample of the net reliability gap formulas (H35, L40)
- The statistics formulas (H42:H47)
- The weighted mean formula (H50)
- Confirm no cells in the target ranges are None or empty.

## Key Cautions
- The avoid-artifact warns about #NAME? errors from invalid function names. Use only standard Excel function names: MIN, MAX, MEDIAN, AVERAGE, PERCENTILE, SUMPRODUCT, SUM, INDEX, MATCH.
- Do NOT use PERCENTILE.INC or PERCENTILE.EXC — use plain PERCENTILE.
- Do NOT use _xlfn. prefixes.
- Inspect the actual workbook structure before writing any formulas. Do not assume column/row positions.

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