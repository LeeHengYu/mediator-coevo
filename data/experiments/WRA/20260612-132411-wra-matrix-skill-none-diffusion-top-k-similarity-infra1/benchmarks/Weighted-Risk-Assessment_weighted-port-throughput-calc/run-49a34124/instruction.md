# Task Instruction

You must update an Excel workbook at `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Follow these steps precisely:

## Step 0: Inspect the workbook
1. `mkdir -p /root/output`
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: print rows 1–55 for columns A–L (values AND formulas). Pay special attention to:
     - Column D rows 12–17, 19–24, 26–31 (series codes)
     - Row 10 columns H–L (years)
     - What is already in H12:L17, H19:L24, H26:L31 (should be empty/yellow)
     - Rows 35–40 (port names and any existing content)
     - Rows 42–47 (stat labels: min, max, median, mean, 25th, 75th percentile)
     - Row 50 (CPA weighted mean)
   - Sheet `Data`: print rows 1–40 focusing on rows 21–38 to understand the data layout (which row has which series code, which columns have which years, etc.)
3. Print cell backgrounds/fills for a few yellow cells to confirm target ranges.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a spreadsheet **formula** (not a computed value) that looks up data from sheet `Data` rows 21:38. The formula must use two keys:
- The series code from column D of that row on sheet `Task`
- The year from row 10 of that column on sheet `Task`

Use one of these patterns: `INDEX/MATCH`, `VLOOKUP/MATCH`, `HLOOKUP/MATCH`, or `XLOOKUP/MATCH`.

IMPORTANT: You must write the formulas as strings starting with `=` into the cells. Use `openpyxl` and set the cell `.value` to the formula string. Make sure:
- References to the Data sheet use the correct sheet name syntax (e.g., `Data!A21:A38` or `Data!$A$21:$A$38`).
- Inspect the Data sheet carefully to determine whether series codes are in a column and years are in a row (or vice versa), and choose the right lookup pattern accordingly.
- Use absolute references (`$`) where needed to allow the formula to be consistent across the range, but relative references for the parts that should shift (the year column and the series code row).
- After writing formulas, re-read a few cells to confirm the formula strings are stored correctly.

## Step 2: Net container flow and summary statistics in H35:L40 and H42:L47

For H35:L40, write formulas that compute:
`(Loaded Containers Inbound - Loaded Containers Outbound) / Terminal Throughput Capacity * 100`

The three data blocks are:
- Loaded Containers Inbound: H12:L17
- Loaded Containers Outbound: H19:L24  
- Terminal Throughput Capacity: H26:L31

So for cell H35: `=(H12-H19)/H26*100`, H36: `=(H13-H20)/H27*100`, etc. Adjust row references for each row, and column references for each column.

For H42:L47, write column-wise summary statistic formulas over H35:L40:
- Row 42: `=MIN(H35:H40)` (adjust column for each)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

IMPORTANT: Check the labels in column D or nearby columns for rows 42–47 to confirm which row corresponds to which statistic. Adjust the mapping if the labels differ from the order above.

## Step 3: Weighted mean in H50:L50

For each column H–L, write a `SUMPRODUCT` formula:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This computes the weighted mean of the net container flow percentages (H35:H40) weighted by terminal throughput capacity (H26:H31).

## Step 4: Save and validate
1. Save to `/root/output/result.xlsx` preserving all existing formatting. Use `openpyxl` with the workbook you opened (do NOT use write-only mode; keep styles intact).
2. Re-open the saved file and print the formula content of representative cells from each range (e.g., H12, L17, H35, L40, H42, H47, H50, L50) to verify formulas are correctly stored.
3. Confirm no extra sheets were added and the original sheets are intact.

## Critical constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting (fonts, fills, borders, etc.).
- Write FORMULAS (strings starting with `=`), not hardcoded values, for all cells.
- Use `openpyxl` throughout. If you need to preserve existing content/formatting, load with `data_only=False` (the default).
- The final file must be at `/root/output/result.xlsx`.

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