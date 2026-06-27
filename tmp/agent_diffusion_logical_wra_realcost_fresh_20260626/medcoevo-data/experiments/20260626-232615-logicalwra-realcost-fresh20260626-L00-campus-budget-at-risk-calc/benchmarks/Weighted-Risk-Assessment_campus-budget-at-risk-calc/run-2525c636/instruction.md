# Task Instruction

Execute the following steps precisely to complete the campus budget workbook task.

## Setup
1. `cp /root/data/workbook.xlsx /root/output/result.xlsx`
2. Install openpyxl if needed: `pip install openpyxl`

## Inspection
3. Open `/root/output/result.xlsx` with openpyxl (data_only=False) and inspect:
   - Sheet names (confirm `Task` and `Data` exist)
   - On `Task` sheet: read row 10 (H10:L10) to see the year headers; read column D rows 12-17, 19-24, 26-31 to see the series codes; read rows 35-40 column D for department names or codes; read H35:L40 to see if there are labels; read row 42-47 column D-G for stat labels (min, max, median, mean, 25th, 75th percentile); read H50:L50 area and nearby labels.
   - On `Data` sheet: read rows 21-38 completely to understand the data layout — identify which row is the header row, what columns contain series codes, and how years map to columns. Also check row 1-5 for any header structure.
   - Print all findings so you understand the exact layout before writing any formulas.

## Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31
4. Based on the inspection, write a Python script using openpyxl to populate the yellow cells. For each cell in the three blocks (rows 12-17, 19-24, 26-31; columns H-L):
   - The formula must use the series code from column D of that row and the year from row 10 of that column.
   - Use INDEX/MATCH pattern referencing Data!$rows21:38 (adjust exact range based on inspection). The formula pattern should be something like:
     `=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))`
   - Make sure the series code column reference uses absolute column ($D12) and the year reference uses absolute row (H$10) so the formula copies correctly across the block.
   - Anchor the Data range references with $ signs appropriately.
   - IMPORTANT: Verify the exact row/column layout on Data sheet before constructing the formula. The data rows are 21:38, but you need to know which column has series codes and which row within that range has year headers.

## Step 2: Net Budget Buffer in H35:L40 and Summary Stats in H42:L47
5. Determine which of the three blocks corresponds to:
   - Committed Funding
   - Operating Spend  
   - Approved Budget Base
   Read the labels in the Task sheet (likely in column B-G near rows 11, 18, 25) to identify which block is which.

6. For H35:L40, enter formulas computing: `(Committed Funding - Operating Spend) / Approved Budget Base * 100`
   - Each cell references the corresponding cell from the three blocks. For example, if Committed Funding is rows 12-17, Operating Spend is rows 19-24, and Approved Budget Base is rows 26-31, then H35 = `=(H12-H19)/H26*100`. Adjust based on actual block assignments.
   - Use relative references that work for each row/column in the 6×5 grid.

7. For H42:L47, enter column-wise summary statistics over H35:L40:
   - Identify which row is which statistic from the labels in column D-G (rows 42-47).
   - Use: `=MIN(H35:H40)`, `=MAX(H35:H40)`, `=MEDIAN(H35:H40)`, `=AVERAGE(H35:H40)`, `=PERCENTILE(H35:H40,0.25)`, `=PERCENTILE(H35:H40,0.75)` — assign each to the correct row based on the label.

## Step 3: Weighted Mean in H50:L50
8. For each cell in H50:L50, enter a SUMPRODUCT formula:
   `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
   This computes the weighted mean of the Net Budget Buffer percentages weighted by Approved Budget Base. Adjust the Approved Budget Base range reference if the block assignment differs.

## Save and Validate
9. Save the workbook (keep_vba=False, no new sheets added).
10. Reopen the saved file and verify:
    - No new sheets were added
    - Cells H12, L17, H19, L24, H26, L31 contain formula strings (not None)
    - Cells H35, L40 contain formula strings
    - Cells H42, L47 contain formula strings
    - Cells H50, L50 contain formula strings
    - Print a sample of formulas from each block for confirmation

## Critical Rules
- Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify any existing formatting, values outside the target cells, or sheet structure.
- Use openpyxl to write Excel formula strings (starting with `=`), NOT computed Python values.
- The formulas must be valid Excel formulas that would compute correctly when opened in Excel.

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