# Task Instruction

Execute the following steps to complete the Excel workbook task.

## Phase 0: Setup
```bash
mkdir -p /root/output
cp /root/data/workbook.xlsx /root/output/result.xlsx
```

## Phase 1: Inspect the workbook structure
Using Python with openpyxl (load_only to read), inspect:
1. On sheet `Task`: read the series codes in column D for rows 12-17, 19-24, 26-31 (these are the three lookup blocks). Read the years in row 10 for columns H through L. Read any labels in column B or C for rows 35-40, 42-47, 50 to understand what goes where.
2. On sheet `Data`: read rows 21-38 to understand the data layout — identify which row contains headers, which column contains series codes, and how the year columns are arranged.
3. Print all findings clearly so you can map series codes and years precisely.

## Phase 2: Write formulas using openpyxl
Open `/root/output/result.xlsx` with openpyxl (not data_only, not read_only). Work only on the `Task` sheet.

### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write an `INDEX(MATCH,MATCH)` formula that:
- Uses the Data sheet's data range (rows 21-38, appropriate columns) as the array
- MATCHes the series code from column D of the current row against the series code column on Data
- MATCHes the year from row 10 against the year header row on Data
- Use exact match (0) for both MATCH functions

Be precise about:
- The Data sheet range for the array argument (exclude the header row and label column from the array, but include them in the MATCH ranges)
- Using absolute references where needed (e.g., $D12 for series code, H$10 for year)
- The sheet reference syntax: `Data!` prefix

### Step 2: Net production slack in H35:L40
Write formulas that compute:
`(Finished Output - Scrap And Rework) / Rated Production Capacity * 100`

Based on your Phase 1 inspection:
- "Finished Output" values are in H12:L17
- "Scrap And Rework" values are in H19:L24  
- "Rated Production Capacity" values are in H26:L31

So for cell H35: `=(H12-H19)/H26*100`, and similarly for the rest of the 6×5 block.

### Step 2b: Summary statistics in H42:L47
For each column (H through L), compute over the 6 values in rows 35-40:
- Row 42: `=MIN(H35:H40)` (or whichever row corresponds to minimum — check the labels from Phase 1)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

**IMPORTANT**: Map each statistic to the correct row by reading the labels in column B/C/D for rows 42-47 during Phase 1. The order above is just a guess — use the actual labels.

### Step 3: Weighted mean in H50:L50
For each column, write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This uses the net production slack percentages as values and Rated Production Capacity as weights.

## Phase 3: Save and verify
1. Save the workbook: `wb.save('/root/output/result.xlsx')` — this is critical, do NOT skip.
2. Reopen the saved file and verify that cells H12, L17, H19, L24, H26, L31, H35, L40, H42, L47, H50, L50 all contain formula strings (not None, not static values).
3. Print the formula content of a sample of cells from each block to confirm correctness.

## Critical Reminders
- Do NOT add new sheets, macros, VBA, or external links.
- Do NOT modify formatting.
- The file MUST be saved to `/root/output/result.xlsx`.
- All target cells must contain Excel formulas, not computed Python values.
- Re-read each cell range from the actual workbook before writing to ensure correct row/column mapping.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=hard, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.