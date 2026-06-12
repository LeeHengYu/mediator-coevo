# Task Instruction

Execute the following steps carefully and in order.

## 0. Environment Setup
```bash
pip install openpyxl
mkdir -p /root/output
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl and print:
- Sheet names
- On sheet `Task`: the contents of cells D12:D17, D19:D24, D26:D31 (series codes), and H10:L10 (years in row 10). Also print cells H35:H40 labels, H42:H47 labels, and H50 label to understand the layout.
- On sheet `Data`: print rows 20-38 fully (all columns with data) so we can see the exact layout — column headers, series codes, and where the year columns start. Also print row 1 or whichever row has column headers.

Print everything with cell coordinates so we know exact positions.

## 2. Understand the Data sheet layout
From the inspection, identify:
- Which column on `Data` sheet contains the series codes (likely column A or B)
- Which row on `Data` sheet contains the year headers
- The exact range of the data table in rows 21:38

## 3. Write formulas into H12:L17, H19:L24, H26:L31
Using openpyxl, write INDEX/MATCH formulas into each yellow cell. The formula pattern for cell H12 (for example) should be:
```
=INDEX(Data!$C$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$C$20:$XX$20, 0))
```
But you MUST adjust the ranges based on what you found in step 1:
- The first argument to INDEX should be the data values range on the Data sheet (excluding the series code column and header row, just the numeric data area in rows 21:38)
- The first MATCH should look up the series code from column D of the current row against the series code column in Data rows 21:38
- The second MATCH should look up the year from row 10 of the Task sheet against the year header row on the Data sheet

IMPORTANT: Use absolute references ($) for the lookup ranges but relative references for the lookup values ($D12 for series code with absolute column, H$10 for year with absolute row). This ensures the formula copies correctly across the 5 columns and 6 rows of each block.

Write the formulas for all three blocks: H12:L17, H19:L24, H26:L31.

## 4. Write Net Patient Flow formulas in H35:L40
For each cell in H35:L40, compute:
```
=(HXX_admissions - HXX_discharges) / HXX_capacity * 100
```
where HXX_admissions is the corresponding cell in H12:L17, HXX_discharges is in H19:L24, and HXX_capacity is in H26:L31.

For example, H35 = (H12 - H19) / H26 * 100
H36 = (H13 - H20) / H27 * 100, etc.
And similarly across columns I through L.

## 5. Write summary statistics in H42:L47
For each column (H through L), in the six rows 42-47, write formulas for:
- Row 42: =MIN(H35:H40)
- Row 43: =MAX(H35:H40)
- Row 44: =MEDIAN(H35:H40)
- Row 45: =AVERAGE(H35:H40)
- Row 46: =PERCENTILE(H35:H40, 0.25)
- Row 47: =PERCENTILE(H35:H40, 0.75)

Check the labels in column D or G for rows 42-47 to confirm the correct order. Adjust if the labels say something different.

## 6. Write weighted mean in H50:L50
For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net Patient Flow using Effective Bed Capacity as weights.

## 7. Save the workbook
Save to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets.

## 8. Validate
Reopen `/root/output/result.xlsx` with openpyxl (data_only=False) and print:
- The formula in H12 to verify it looks correct
- The formula in H35
- The formula in H42
- The formula in H50

Then reopen with an xlsx evaluation library or just confirm the formulas are syntactically correct by checking they start with '=' and reference the correct sheets.

Also verify that no cells in the target ranges are None — every cell should contain a formula string.

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