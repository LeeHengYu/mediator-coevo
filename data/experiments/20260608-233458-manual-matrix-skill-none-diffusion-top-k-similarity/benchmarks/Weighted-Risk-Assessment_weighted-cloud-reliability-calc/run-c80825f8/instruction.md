# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup & Inspection

```bash
mkdir -p /root/output
pip install openpyxl
```

Open and inspect `/root/data/workbook.xlsx` using openpyxl to understand:
- Sheet names (expect `Task` and `Data`)
- On sheet `Task`: what is in column D rows 12-17, 19-24, 26-31 (series codes), and what is in row 10 columns H-L (years)
- On sheet `Data`: the structure of rows 21-38 (what columns hold series codes, and where years appear)
- On sheet `Task`: what is in H35:L40, H42:L47, H50:L50 currently (empty or has content)
- On sheet `Task`: what labels are in rows 35-40 (the six regions), rows 42-47 (min/max/median/mean/25th/75th), row 50 (GCM)
- Identify the exact row/column layout of the Data sheet so you know what VLOOKUP/INDEX-MATCH formulas to write

Print all of this information before proceeding.

## 1. Populate H12:L17, H19:L24, H26:L31 with lookup formulas

For each cell in these three blocks, write a spreadsheet formula (not a Python-computed value) that:
- Takes the series code from column D of the same row on sheet `Task`
- Takes the year from row 10 of the same column on sheet `Task`
- Looks up the value from sheet `Data` rows 21:38

Use INDEX-MATCH or VLOOKUP-MATCH pattern. The exact formula depends on the Data sheet layout you discovered in step 0. For example, if Data has series codes in column A and years in a header row, an INDEX-MATCH-MATCH formula would be appropriate:
`=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))`

Adjust the ranges based on the actual layout discovered. The key requirement is that the formula uses one of the allowed patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.

IMPORTANT: When writing formulas with openpyxl, use `Translator` or manually write the formula string for each cell. Make sure row references that should be absolute use `$` signs appropriately (column D reference should lock the column `$D`, row 10 reference should lock the row `$10`).

## 2. Populate H35:L40 with Net reliability gap

The formula for each cell in H35:L40 is:
`(Successful API Requests - Failed API Requests) / Compute Capacity * 100`

Based on the three blocks:
- H12:L17 = first metric block (check which one is Successful API Requests, Failed API Requests, or Compute Capacity)
- H19:L24 = second metric block
- H26:L31 = third metric block

Identify which block corresponds to which metric by reading the labels on the Task sheet (likely around rows 11, 18, 25). The six rows in each block correspond to the six regions.

For example, if block 1 (rows 12-17) is Successful API Requests, block 2 (rows 19-24) is Failed API Requests, and block 3 (rows 26-31) is Compute Capacity, then:
`H35 = (H12 - H19) / H26 * 100`

Adjust based on actual layout. Write these as spreadsheet formulas.

## 3. Populate H42:L47 with summary statistics

For each column H through L:
- Row 42 (minimum): `=MIN(H35:H40)`
- Row 43 (maximum): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`

Check the labels in column D (or nearby) for rows 42-47 to confirm which row is which statistic. Assign formulas accordingly.

## 4. Populate H50:L50 with weighted mean using SUMPRODUCT

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net reliability gap values (H35:H40) weighted by Compute Capacity (H26:H31). Adjust the Compute Capacity range if it's in a different block.

## 5. Preserve formatting

When opening the workbook with openpyxl, do NOT use `data_only=True`. Open it normally so formulas are preserved. When saving, the formatting should be preserved automatically as long as you only modify cell values (formulas) and don't touch styles, dimensions, or other properties.

## 6. Save

Save the workbook to `/root/output/result.xlsx`.

## 7. Verification

After saving, reopen `/root/output/result.xlsx` and verify:
- Cells H12:L17, H19:L24, H26:L31 contain formula strings (not None/empty)
- Cells H35:L40 contain formula strings
- Cells H42:L47 contain formula strings
- Cells H50:L50 contain formula strings
- The formulas reference the correct cells/ranges
- No new sheets were added
- Print a sample of formulas from each block for confirmation

Do all work in a single Python script. Print diagnostic information at each step so issues can be caught early.

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