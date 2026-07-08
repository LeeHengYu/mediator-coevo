# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Pre-work
1. `mkdir -p /root/output`
2. Open and inspect `/root/data/workbook.xlsx` with openpyxl to confirm:
   - Sheet names (expect 'Task' and 'Data').
   - On 'Task': read row 10 to find the years in columns H–L; read column D rows 12–17, 19–24, 26–31 to find the series codes; read row labels in H35:H40 area and H42:H47 area to understand the layout.
   - On 'Data': read rows 21–38 to understand the data table structure (which row has headers, which column has series codes, where years appear).
   Print all discovered values so the formula references are correct.

## Step 1 – Lookup formulas in yellow cells
For each cell in H12:L17, H19:L24, and H26:L31, write an INDEX/MATCH formula that:
- Uses the series code from column D of that row and the year from row 10 of that column.
- Looks up from sheet Data rows 21:38.
- Pattern: `=INDEX(Data!<data_range>, MATCH(D<row>, Data!<series_code_column>, 0), MATCH(<year_cell>, Data!<year_header_row>, 0))`
- Adjust the exact ranges based on what you discover in the inspection step. The MATCH for the series code should search the column in Data that contains series codes; the MATCH for the year should search the row in Data that contains year headers.
- Use absolute references for the data range and lookup arrays so they don't shift when filled across cells.

## Step 2 – Net reliability gap (H35:L40)
For each of the 6 regions (rows 35–40) and each year column (H–L), write a formula:
`=(H12 - H19) / H26 * 100`
where H12 is the Successful API Requests cell, H19 is the Failed API Requests cell, and H26 is the Compute Capacity cell for that region/year. Adjust row references for each region row.

## Step 2 continued – Statistics (H42:L47)
For each year column H–L, in the 6 statistic rows write:
- Minimum: `=MIN(H35:H40)` (adjust column)
- Maximum: `=MAX(H35:H40)`
- Median: `=MEDIAN(H35:H40)`
- Simple mean: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40, 0.25)` — use classic PERCENTILE, NOT PERCENTILE.INC or PERCENTILE.EXC to avoid #NAME? errors
- 75th percentile: `=PERCENTILE(H35:H40, 0.75)`

IMPORTANT: Check the actual row labels in H42:H47 to determine which row gets which statistic. Map them correctly.

## Step 3 – Weighted mean (H50:L50)
For each year column H–L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
This computes the weighted mean of the Net reliability gap percentages using Compute Capacity as weights.

## Final steps
1. After writing all formulas, save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and spot-check a few cells to confirm formulas are present (not just values) and are syntactically correct.
3. Verify no extra sheets were added and no formatting was altered.

## Critical constraints
- Use `data_only=False` when loading so existing formulas are preserved.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change any existing formatting.
- Use classic `PERCENTILE` function, never `PERCENTILE.INC` or `PERCENTILE.EXC`.
- Inspect the actual sheet layout FIRST before writing any formulas — do not assume row/column positions.

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