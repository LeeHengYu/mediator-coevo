# Task Instruction

Execute the following steps in order to produce /root/output/result.xlsx.

## 0 – Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1 – Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and print:
- Sheet names.
- On sheet `Task`: the contents of row 10 (columns A–S), column D for rows 12–17, 19–24, 26–31, 35–40, and the labels in column D (or wherever they are) for rows 42–47 and row 50.
- On sheet `Data`: the contents of row 20 (header row above the data block) and column A or B for rows 21–38, plus the first data row to confirm orientation.

This tells you the exact header row, ID column, and data range on `Data`, and the exact label order for the summary statistics on `Task`.

## 2 – Write lookup formulas (Step 1)
Using openpyxl, write INDEX/MATCH formulas into the yellow cells.

For every cell in the three blocks H12:L17, H19:L24, H26:L31 use a formula like:

```
=INDEX(Data!$C$21:$S$38, MATCH($D12, Data!$B$21:$B$38, 0), MATCH(H$10, Data!$C$20:$S$20, 0))
```

**Important adjustments** (confirm with the inspection output):
- The data range columns (e.g., $C$21:$S$38) must start at the first numeric column on `Data` and span all years.
- The row-lookup vector (e.g., Data!$B$21:$B$38) must be the column containing the series codes that match column D on `Task`.
- The column-lookup vector (e.g., Data!$C$20:$S$20) must be the header row containing years that match row 10 on `Task`.
- Use $ anchoring: $D12 (column absolute, row relative) and H$10 (column relative, row absolute) so the formula copies correctly across the block.
- Adjust the row references (e.g., $D12 becomes $D19 for the second block, $D26 for the third) for each block.

## 3 – Net reliability gap (Step 2, rows 35–40)
For each cell in H35:L40, write a formula that computes:
```
=(H12 - H19) / H26 * 100
```
where H12 is the Successful API Requests value, H19 is the Failed API Requests value, and H26 is the Compute Capacity value for the same region and year. Adjust row references per region (row 35 uses rows 12,19,26; row 36 uses 13,20,27; etc.).

Confirm that the three blocks (rows 12–17, 19–24, 26–31) correspond to Successful API Requests, Failed API Requests, and Compute Capacity respectively by checking the block headers on `Task`.

## 4 – Summary statistics (Step 2, rows 42–47)
Read the labels in column D (or wherever) for rows 42–47 from the inspection output. Then write the corresponding column-wise formulas over H42:L47. The six statistics are MIN, MAX, MEDIAN, AVERAGE (simple mean), 25th percentile, 75th percentile — but place them in the order the labels appear. For example if row 42 says "Minimum":
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- Mean: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)`

Match each formula to the label in that row.

## 5 – Weighted mean (Step 3, row 50)
For each cell in H50:L50 write:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net reliability gap percentages using Compute Capacity as weights.

## 6 – Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets or macros.

## 7 – Verify
Reopen the saved file with openpyxl (data_only=False) and print a sample of cells from each block (e.g., H12, H19, H26, H35, H42, H50) to confirm formulas were written correctly. Check that no cells in the target ranges are None or empty.

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