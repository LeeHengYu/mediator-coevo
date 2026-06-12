# Task Instruction

Execute the following steps in order to produce /root/output/result.xlsx.

Step 0 – Inspect the workbook
```
import openpyxl, pprint
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for s in wb.sheetnames:
    print(f'--- Sheet: {s} ---')
    ws = wb[s]
    print(f'  Dimensions: {ws.dimensions}')
    # Print first 50 rows to understand layout
    for row in ws.iter_rows(min_row=1, max_row=50, values_only=False):
        vals = [(c.coordinate, c.value) for c in row if c.value is not None]
        if vals:
            print(vals)
```
Read the output carefully. In particular note:
- The exact series codes in column D of the Task sheet (rows 12-17, 19-24, 26-31, 35-40).
- The years in row 10 of the Task sheet (columns H-L).
- The layout of the Data sheet rows 21-38 (which column holds the series code, which row/column holds the year headers, and where the numeric data lives).
- The labels in rows 42-47 (min, max, median, mean, 25th, 75th percentile) and row 50 (CPA weighted mean).
- The port names / order in the three lookup blocks and the net-flow block.

Step 1 – Write lookup formulas into H12:L17, H19:L24, H26:L31
Using openpyxl, open the workbook (with data_only=False so existing formulas are preserved) and write INDEX/MATCH formulas into each yellow cell.

For a cell at row r, column c (where c maps to H=8, I=9, J=10, K=11, L=12), the formula pattern is:

=INDEX(Data!$B$21:$Z$38, MATCH($D{r}, Data!$A$21:$A$38, 0), MATCH({col}$10, Data!$B$20:$Z$20, 0))

BUT first verify from Step 0 output:
- Which column on Data holds the series codes (likely column A) and which row holds the year headers (likely row 20 or row 21). Adjust the ranges accordingly.
- Which columns span the data (B? C? through what?).

Use mixed references: $D with the actual row number for the series code, and the column letter with $10 for the year. For example, cell H12 would be:
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))

Write these formulas for all 60 cells (3 blocks × 6 rows × 5 columns).

Step 2 – Net container flow formulas in H35:L40
For each cell at row r, column c, write:
=(H{inbound_row} - H{outbound_row}) / H{capacity_row} * 100

where inbound_row, outbound_row, capacity_row correspond to the same port in the three blocks above (rows 12-17 for block 1, 19-24 for block 2, 26-31 for block 3). Determine which block is Loaded Containers Inbound, which is Loaded Containers Outbound, and which is Terminal Throughput Capacity from the block headers visible in Step 0. The six ports in rows 35-40 should match the same order as in the lookup blocks.

For example, if block 1 (rows 12-17) = Inbound, block 2 (rows 19-24) = Outbound, block 3 (rows 26-31) = Capacity, then for port 1 in H35:
=(H12 - H19) / H26 * 100

Step 3 – Summary statistics in H42:L47
For each column (H through L), write:
- MIN row:   =MIN(H35:H40)
- MAX row:   =MAX(H35:H40)
- MEDIAN row: =MEDIAN(H35:H40)
- MEAN row:  =AVERAGE(H35:H40)
- 25th pctl:  =PERCENTILE(H35:H40, 0.25)
- 75th pctl:  =PERCENTILE(H35:H40, 0.75)

Match the label order from Step 0 output (rows 42-47) to the correct function.

Step 4 – Weighted mean in H50:L50
For each column c (H-L):
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)

This uses the net-flow percentages as values and Terminal Throughput Capacity as weights.

Step 5 – Save
```python
import os
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

Step 6 – Verify
Reload the saved file and print the formulas in a sample of cells (e.g., H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50) to confirm they are correctly written and not None.

Also open the file with data_only=True (if the runtime supports it) or use a formula-checking approach to ensure no cells in the target ranges are None/empty.

IMPORTANT NOTES:
- Do NOT use data_only=True when loading the workbook for editing; that strips formulas.
- Do NOT add any new sheets, macros, VBA, or external links.
- Preserve all existing formatting — only write to the specified cell ranges.
- If Step 0 reveals that the Data sheet layout differs from assumptions (e.g., different column for series codes, different row for year headers), adjust all formula ranges accordingly before writing.

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