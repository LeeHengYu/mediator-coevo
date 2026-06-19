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
- On sheet `Task`: read row 10 (especially H10:L10) to see the year headers; read column D rows 12-31 to see the series codes; read the layout of rows 35-50 to understand what labels are there.
- On sheet `Data`: read rows 21-38 to understand the data layout — identify which column holds the series codes, which row holds years, and where the numeric data lives. Print out the first few columns and rows so you understand the exact structure (column A through at least column Z of rows 20-40).

Print all of this information before making any edits.

## 2. Populate H12:L17, H19:L24, H26:L31 with lookup formulas

For each cell in the ranges H12:L17, H19:L24, and H26:L31 on sheet `Task`:
- The two inputs are: (a) the series code from column D of that row, and (b) the year from row 10 of that column.
- The lookup source is sheet `Data` rows 21:38.
- Use INDEX/MATCH formulas. The exact formula pattern depends on the Data sheet layout you discovered in step 1. A typical pattern would be:
  `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
  but you MUST adjust the ranges based on what you actually see in the Data sheet (which column has series codes, which row has year headers, and where the data block starts/ends).

Use openpyxl to write these formulas as strings (not computed values). Make sure:
- Row references use `$` for the lookup arrays but the row/column of the current cell varies appropriately.
- The series code reference locks the column ($D) but not the row.
- The year reference locks the row ($10) but not the column.

## 3. Populate H35:L40 with Net reliability gap formulas

The Net reliability gap formula is:
`(Successful API Requests - Failed API Requests) / Compute Capacity * 100`

Based on the layout:
- H12:L17 = first block (one of the three indicators)
- H19:L24 = second block
- H26:L31 = third block

You need to figure out which block corresponds to which indicator by reading the labels in column B or C near rows 11, 18, 25. Then map:
- Successful API Requests block rows
- Failed API Requests block rows  
- Compute Capacity block rows

For each cell in H35:L40, write a formula like:
`=(H12-H19)/H26*100`
adjusting row references to match the correct rows for each region (rows 35-40 correspond to the 6 regions in the same order as they appear in each block).

## 4. Populate H42:L47 with summary statistics

For each column H through L:
- H42 (or whichever row is MIN): `=MIN(H35:H40)`
- H43 (MAX): `=MAX(H35:H40)`
- H44 (MEDIAN): `=MEDIAN(H35:H40)`
- H45 (MEAN): `=AVERAGE(H35:H40)`
- H46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- H47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

IMPORTANT: Check the actual labels in column B/C/D of rows 42-47 to determine which row gets which function. The order listed above is just a guess — match the actual labels (minimum, maximum, median, mean/average, 25th percentile, 75th percentile).

## 5. Populate H50:L50 with weighted mean using SUMPRODUCT

For each column (H through L), write:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net reliability gap values (H35:H40) weighted by Compute Capacity (H26:H31).

## 6. Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change any formatting, do NOT add sheets.

## 7. Verify
Reopen `/root/output/result.xlsx` and print out the formulas in a sample of cells (e.g., H12, L17, H35, H42, H50) to confirm they are correctly written formula strings. Also confirm sheet names are unchanged and no extra sheets exist.

## Critical Notes
- Use `openpyxl` with `load_workbook(filename, keep_vba=False)` — do NOT use `data_only=True` when loading since you need to preserve and write formulas.
- When writing formulas, they must be strings starting with `=`.
- Do NOT overwrite any cells outside the specified ranges.
- Do NOT change formatting (fonts, colors, borders, etc.).
- Inspect before editing — read the actual cell contents and layout first, then plan the exact formulas based on what you see.

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