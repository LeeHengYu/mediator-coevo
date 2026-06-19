# Task Instruction

Execute the following steps exactly, in order.

## 0. Environment Setup
```bash
pip install openpyxl
mkdir -p /root/output
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and print:
- Sheet names
- Sheet `Task`: cells D12:D17, D19:D24, D26:D31 (series codes), row 10 columns H-L (years), cells H35:L40 labels/content, H42:L47 labels, H50:L50 label, and any content in the weight row or CPA row.
- Sheet `Data`: rows 21-38, first 15 columns — print all values so we can see the data layout (column headers, series codes, year columns).
- Also print the exact column letters and row numbers of the Data sheet header row (row 20 or wherever headers are).

This inspection is critical. Do NOT skip it.

## 2. Build formulas in Python using openpyxl
Write a Python script that:

### 2a. Lookup formulas (H12:L17, H19:L24, H26:L31)
For each cell in these three blocks, write a formula that looks up the value from sheet `Data` rows 21:38 using the series code in column D of the current row and the year in row 10. Use INDEX/MATCH pattern:
```
=INDEX(Data!<data_range>, MATCH(D<row>, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```
Adjust the exact ranges based on what you found in step 1. The `<data_range>` should cover the numeric portion of rows 21:38. The `<series_code_column>` is the column in Data that contains the series codes. The `<year_header_row>` is the row in Data that contains the year headers.

Make sure:
- Row references for D<row> and H$10 (or I$10, J$10, etc.) are correct for each cell.
- The column letter changes as you move from H to L.
- Use absolute references ($) where needed to keep the lookup ranges fixed.

### 2b. Net container flow (H35:L40)
For each of the 6 ports (rows 35-40), calculate:
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to Loaded Containers Inbound (rows 12-17), H19 to Loaded Containers Outbound (rows 19-24), and H26 to Terminal Throughput Capacity (rows 26-31). Adjust row references so that row 35 uses rows 12, 19, 26; row 36 uses 13, 20, 27; etc. Adjust column letters for H through L.

### 2c. Statistics (H42:L47)
For each column H through L, in rows 42-47:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE.INC(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE.INC(H35:H40,0.75)` (75th percentile)

**CRITICAL**: Use exactly `PERCENTILE.INC` — not `PERCENTILE`, not `PERCENTILE.EXC`, not any other variant. The previous run failed because of #NAME? errors on percentile functions. Double-check the spelling.

### 2d. Weighted mean for CPA (H50:L50)
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of net container flow percentages using Terminal Throughput Capacity as weights.

### 2e. Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change any formatting, do NOT add sheets, macros, or VBA.

## 3. Verify
After saving, reopen `/root/output/result.xlsx` with openpyxl (data_only=False) and print:
- A sample of lookup formulas (H12, L17, H19, L24, H26, L31)
- Net flow formulas (H35, L40)
- All statistics formulas (H42:L47)
- Weighted mean formulas (H50:L50)

Confirm:
- No #NAME? candidates: search all formula strings for misspellings
- PERCENTILE.INC is spelled exactly right
- All cell references look correct

## 4. Verify labels match expectations
Check that the labels in column G (or wherever they are) for rows 42-47 match: minimum, maximum, median, mean, 25th percentile, 75th percentile — and confirm the order matches the formulas you placed. If the order in the sheet is different (e.g., row 42 is mean, not min), adjust your formulas to match the labels.

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