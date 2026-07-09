# Task Instruction

Execute the following steps carefully to produce `/root/output/result.xlsx`.

## 0 – Preparation

```bash
mkdir -p /root/output
pip install openpyxl
```

Open and inspect the workbook:
```python
import openpyxl
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for s in wb.sheetnames:
    print(s)
ws_task = wb['Task']
ws_data = wb['Data']
```

Inspect the Task sheet layout:
- Print rows 10-50 for columns D through L to understand the structure (series codes in column D, years in row 10, yellow target ranges, region names, etc.).
- Print the Data sheet rows 21-38 to understand the source data layout (which row/column holds what).

Capture and print:
- The exact series codes in D12:D17, D19:D24, D26:D31 (these should correspond to the three indicator blocks).
- The years in H10:L10.
- The structure of Data!21:38 (header row, data rows, column layout).
- The region names in D35:D40 and any labels in D42:D47.

## 1 – Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31

For each yellow cell, write a formula that looks up the value from the Data sheet using the series code (column D of the same row on Task sheet) and the year (row 10 of the same column on Task sheet).

Use `INDEX/MATCH` pattern. The Data sheet source range is rows 21:38. Determine:
- Which column in the Data sheet contains the series codes (inspect to find it).
- Which row in the Data sheet contains the years (inspect to find it).

The formula pattern for cell H12 should be something like:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```

Adjust the exact ranges based on what you find in the Data sheet. The `$D12` locks the column, `H$10` locks the row, so the formula can be applied across the block.

Write formulas (not hardcoded values) into every cell in H12:L17, H19:L24, and H26:L31.

## 2 – Step 2: Net reliability gap in H35:L40 and statistics in H42:L47

The three blocks from Step 1 correspond to three indicators. Based on the task description:
- One block is "Successful API Requests"
- One block is "Failed API Requests"
- One block is "Compute Capacity"

Identify which block (rows 12-17, 19-24, 26-31) corresponds to which indicator by reading the labels on the Task sheet (likely in rows 11, 18, 25 or nearby).

For H35:L40, the formula is:
```
=(SuccessfulAPIRequests - FailedAPIRequests) / ComputeCapacity * 100
```
where each term references the corresponding cell in the appropriate block above. For example, if rows 12-17 are Successful, rows 19-24 are Failed, rows 26-31 are Compute Capacity, then H35 = (H12 - H19) / H26 * 100. Adjust based on actual layout.

For H42:L47 (column-wise statistics over H35:L40):
- Row 42: MIN  → `=MIN(H35:H40)`
- Row 43: MAX  → `=MAX(H35:H40)`
- Row 44: MEDIAN → `=MEDIAN(H35:H40)`
- Row 45: AVERAGE → `=AVERAGE(H35:H40)`
- Row 46: 25th percentile → `=PERCENTILE(H35:H40, 0.25)`
- Row 47: 75th percentile → `=PERCENTILE(H35:H40, 0.75)`

**CRITICAL**: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`). Cross-task feedback shows that `PERCENTILE.INC` causes #NAME? errors in the validation engine. Use the short form `PERCENTILE(range, k)`.

Verify the labels in D42:D47 to confirm which row is which statistic, and match accordingly.

## 3 – Step 3: Weighted mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net reliability gap percentages (H35:H40) weighted by Compute Capacity (H26:H31).

## 4 – Save

Save as `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets.

```python
wb.save('/root/output/result.xlsx')
```

## 5 – Validation

Reload the saved file and print the formula contents (not computed values) of representative cells:
- H12, L17 (lookup block)
- H35, L40 (net reliability gap)
- H42, H46, H47 (statistics – especially percentile rows)
- H50, L50 (weighted mean)

Confirm:
- All formulas reference the correct sheets and ranges.
- No #NAME?, #REF!, or #VALUE! errors in formula text.
- PERCENTILE is used (not PERCENTILE.INC).
- No hardcoded numeric values where formulas are required.
- The workbook has exactly the same sheets as the original.

If any issues are found, fix and re-save before finishing.

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