# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl and inspect:
- Sheet names (confirm `Task` and `Data` exist)
- On sheet `Task`: read row 10 (the year headers in columns H–L), read column D rows 12–17, 19–24, 26–31 (the series codes), read labels in column A or B for rows 12–17, 19–24, 26–31 to understand the three blocks
- On sheet `Data`: read rows 21–38 to understand the data layout — identify which row holds headers, which column holds series codes, and which columns/rows hold years and values
- Read rows 35–40 labels (the six regions), row 42–47 labels (min, max, median, mean, 25th, 75th percentile), and row 50 label
- Print all of this so you understand the exact layout before writing any formulas

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31 (Step 1)

For each cell in these three blocks, write a spreadsheet formula (not a Python-computed value) using one of the allowed lookup patterns. The formula must use two inputs:
- The series code from column D of the same row on sheet `Task`
- The year from row 10 of the same column on sheet `Task`

The data source is sheet `Data` rows 21:38.

Based on your inspection of the Data sheet layout, choose the most appropriate pattern. For example, if Data has series codes in a column and years across a row header, `INDEX(MATCH, MATCH)` is natural:
```
=INDEX(Data!<value_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```

Adjust the exact ranges based on what you find during inspection. Use absolute references for the lookup arrays (mixed references: lock the column for series codes with `$D12`, lock the row for years with `H$10`) so formulas can be applied across the block.

IMPORTANT: When writing formulas with openpyxl, set `cell.value = '=INDEX(...)'` as a string starting with `=`. Do NOT use `data_only` mode. Make sure the workbook is opened without `data_only=True`.

## 3. Net reliability gap formulas in H35:L40 (Step 2, part 1)

For each of the six regions (rows 35–40) and each year column (H–L), write a formula:
```
=(H12 - H19) / H26 * 100
```
where H12 is the Successful API Requests cell, H19 is the Failed API Requests cell, and H26 is the Compute Capacity cell for the same region and year. Adjust row references for each region (row 35 uses rows 12,19,26; row 36 uses rows 13,20,27; etc.).

Verify by checking that the region labels in rows 35–40 match the region labels in rows 12–17 (and 19–24, 26–31).

## 4. Summary statistics in H42:L47 (Step 2, part 2)

For each year column (H–L), write these formulas:
- Row 42 (Minimum): `=MIN(H35:H40)`
- Row 43 (Maximum): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)`

Check the row labels to confirm the correct order (min/max/median/mean/25th/75th may differ — match whatever labels are in column A/B/C).

## 5. Weighted mean in H50:L50 (Step 3)

For each year column (H–L), write:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net reliability gap percentages using Compute Capacity as weights.

## 6. Save

Save the workbook to `/root/output/result.xlsx`. Do NOT change any existing formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.

## 7. Verification

Reopen `/root/output/result.xlsx` with openpyxl and:
- Confirm sheets are only `Task` and `Data` (no extra sheets)
- Confirm cells H12:L17, H19:L24, H26:L31 contain formula strings (start with `=`)
- Confirm cells H35:L40 contain formula strings
- Confirm cells H42:L47 contain formula strings
- Confirm cells H50:L50 contain formula strings
- Print a sample of formulas from each block to verify correctness
- Confirm no data_only artifacts or computed values replaced formulas

## Critical Notes
- You MUST inspect the actual workbook layout before writing formulas. Do not assume row/column positions.
- All cells must contain Excel formulas, not Python-computed values.
- The lookup formulas must reference sheet `Data` with the `Data!` prefix.
- Use `openpyxl` to read and write. Open without `data_only=True`.
- Preserve all existing content and formatting.

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