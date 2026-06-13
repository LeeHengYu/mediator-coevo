# Task Instruction

Execute the following steps in a single Python script to produce /root/output/result.xlsx.

## Preparation
1. `mkdir -p /root/output`
2. Read `/root/data/workbook.xlsx` with openpyxl (keep formatting: `load_workbook('/root/data/workbook.xlsx')`).
3. Inspect the `Task` sheet to confirm layout:
   - Row 10 contains years in columns H–L (columns 8–12).
   - Column D (column 4) contains series codes for rows 12–17, 19–24, 26–31.
   - `Data` sheet rows 21–38 contain the source data.
   - Yellow cells to fill: H12:L17, H19:L24, H26:L31.
   - Net capacity headroom: H35:L40.
   - Statistics: H42:L47.
   - Weighted mean: H50:L50.
4. Print the contents of Task rows 10–50 (cols D, H–L) and Data rows 20–39 (all used cols) so you can verify the layout before writing formulas.

## Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these ranges, write an INDEX/MATCH formula that:
- Looks up the series code from column D of the same row against column A (or whichever column holds series codes) of `Data!$21:$38`.
- Looks up the year from row 10 of the same column against the header row of the Data sheet.
- Pattern: `=INDEX(Data!$B$21:$S$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$S$20,0))`
  Adjust the column ranges ($B$21:$S$38, $A$21:$A$38, $B$20:$S$20) to match what you observe in the Data sheet. The key references are:
  - `$D12` – series code, absolute column, relative row (use `$D` + current row).
  - `H$10` – year, relative column, absolute row 10.
  - Data ranges must be absolute ($).

**Important**: After inspecting the Data sheet, determine the exact column range for data values and for series codes. Use those exact references.

## Step 2 – Net capacity headroom (H35:L40)
Rows 35–40 correspond to the six hospital clusters. The formula for each cell is:
`=(H12 - H19) / H26 * 100`
where the row offsets map as follows:
- Row 35 uses rows 12, 19, 26 (cluster 1)
- Row 36 uses rows 13, 20, 27 (cluster 2)
- Row 37 uses rows 14, 21, 28 (cluster 3)
- Row 38 uses rows 15, 22, 29 (cluster 4)
- Row 39 uses rows 16, 23, 30 (cluster 5)
- Row 40 uses rows 17, 24, 31 (cluster 6)

So for cell (r, c) where r is 35–40 and c is H–L:
`=(H{r-23} - H{r-16}) / H{r-9} * 100`

Write the formula string accordingly, e.g. for row 35, col H: `=(H12-H19)/H26*100`

## Step 3 – Summary statistics (H42:L47)
For each column H–L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

**Critical**: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`). openpyxl does not translate `.INC`/`.EXC` variants and they will produce #NAME? errors. Verify the exact row assignments by checking what labels are in column D/E/F/G of rows 42–47 (min, max, median, mean, 25th, 75th). Adjust the row-to-function mapping to match the actual labels.

## Step 4 – Weighted mean (H50:L50)
For each column H–L:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

## Final
- Save as `/root/output/result.xlsx` using `wb.save('/root/output/result.xlsx')`.
- Re-open the saved file and print the formulas in all modified cells to confirm they are present and syntactically correct.
- Especially verify that rows 46–47 use `PERCENTILE` (no dot suffix) to avoid #NAME? errors.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting.

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