# Task Instruction

Complete the following task to update an Excel workbook with formulas.

## Setup
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx` first, then work on the copy.
2. Install `openpyxl` if not already available (`pip install openpyxl`).

## Inspection Phase (CRITICAL — do this before writing any formulas)
Open `/root/output/result.xlsx` with openpyxl and inspect thoroughly:
- **Sheet `Task`**: Print the contents of rows 10-50, columns A through L. Pay special attention to:
  - Row 10 (years in columns H through L)
  - Column D rows 12-17 (series codes for first block)
  - Column D rows 19-24 (series codes for second block)
  - Column D rows 26-31 (series codes for third block)
  - What labels are in column A or B for rows 12-17, 19-24, 26-31 (to understand which block is Committed Funding, Operating Spend, Approved Budget Base)
  - Rows 35-40 (department names/labels for Net budget buffer)
  - Rows 42-47 (labels for min, max, median, mean, 25th percentile, 75th percentile)
  - Row 50 (Campus Budget Council weighted mean label)
- **Sheet `Data`**: Print rows 21-38 fully (all columns) to understand the data layout — identify which row/column holds series codes, which holds years, and the data orientation (whether data is arranged with years in columns or rows).

Print all of this information before proceeding. You MUST understand the exact layout before writing any formulas.

## Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31

For each yellow cell in these three blocks, write a spreadsheet formula (as a string starting with `=`) that looks up data from `Data!$21:$38`. Each formula must use:
- The series code from column D of the current row on `Task` sheet
- The year from row 10 of the `Task` sheet
- One of these patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH

Based on your inspection of the Data sheet layout, choose the appropriate lookup pattern. If data in rows 21:38 is arranged with series codes in one column and years across columns (or vice versa), adapt accordingly.

For example, if Data has series codes in column A and years across the top row of that range, an INDEX/MATCH formula might look like:
`=INDEX(Data!$A$21:$Z$38,MATCH(D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$A$21:$Z$21,0))`

Adjust column/row references based on actual inspection. Make sure to anchor references appropriately (mixed references) so the formula can be applied across the H:L columns and down the rows correctly.

## Step 2: Net Budget Buffer (H35:L40) and Summary Statistics (H42:L47)

First identify which of the three blocks from Step 1 corresponds to:
- Committed Funding
- Operating Spend  
- Approved Budget Base

Then for H35:L40, write formulas: `=(CommittedFunding - OperatingSpend) / ApprovedBudgetBase * 100` referencing the corresponding cells from the three blocks above. There should be 6 departments × 5 years.

For H42:L47, write column-wise summary statistics over H35:L40:
- Row for minimum: `=MIN(H35:H40)` (and similarly for columns I-L)
- Row for maximum: `=MAX(H35:H40)`
- Row for median: `=MEDIAN(H35:H40)`
- Row for mean: `=AVERAGE(H35:H40)`
- Row for 25th percentile: `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- Row for 75th percentile: `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Match each formula to the correct row based on the labels you observed in the inspection.

## Step 3: Weighted Mean in H50:L50

For each column H through L, write a SUMPRODUCT formula that computes the weighted mean:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

Here H35:H40 are the Net budget buffer percentages (values) and H26:H31 are the Approved Budget Base amounts (weights). Adjust the Approved Budget Base range if your inspection shows it's in a different block.

## Implementation Notes
- Use openpyxl to write formulas as strings (e.g., `ws['H12'] = '=INDEX(...)'`).
- Do NOT overwrite any existing content outside the specified cell ranges.
- Do NOT change formatting, add sheets, macros, VBA, external links, or helper tabs.
- Save the workbook to `/root/output/result.xlsx`.
- After writing all formulas, re-open and print the formula strings in a few sample cells to verify they were written correctly.
- Make sure directory `/root/output/` exists before saving.

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