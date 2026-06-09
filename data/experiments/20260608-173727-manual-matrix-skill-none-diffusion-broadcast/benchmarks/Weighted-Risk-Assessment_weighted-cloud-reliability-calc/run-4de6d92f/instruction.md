# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 – Preparation
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1 – Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl and inspect:
- Sheet `Task`: read row 10 to see the year headers in columns H–L. Read column D rows 12–17, 19–24, 26–31 to see the series codes. Read the labels in column A/B/C for rows 35–50 to understand the layout (region names for rows 35–40, stat labels for rows 42–47, GCM label for row 50). Print all of this so you understand the structure.
- Sheet `Data`: read rows 21–38 to understand the data layout (which row has headers, which column has series codes, where the year columns are). Print the first row (row 21) and a sample data row to see the structure.

## 2 – Write the Python script
Write a single Python script that:

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write an INDEX/MATCH formula that:
- Uses the series code from column D of the same row (e.g., `$D12`)
- Uses the year from row 10 of the same column (e.g., `H$10`)
- Looks up in `Data!$A$21:$A$38` (or wherever the series codes are) for the row match
- Looks up in `Data!$A$21:$XX$21` (or wherever the year headers are) for the column match
- Returns the value from the data range on sheet `Data` rows 21:38

Use the exact column letters and row numbers you discovered in step 1. The formula pattern should be:
`=INDEX(Data!$A$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$A$21:$XX$21, 0))`
Adjust the range references to match the actual data layout.

### Step 2 – Net reliability gap in H35:L40
For each cell in H35:L40, write a formula:
`=(H12 - H19) / H26 * 100`
where H12 corresponds to the Successful API Requests row, H19 to Failed API Requests, and H26 to Compute Capacity for the same region and year. Adjust row references per region (rows 12–17 map to 35–40, rows 19–24 map to 35–40, rows 26–31 map to 35–40). Specifically:
- Row 35: `=(H12-H19)/H26*100`
- Row 36: `=(H13-H20)/H27*100`
- Row 37: `=(H14-H21)/H28*100`
- Row 38: `=(H15-H22)/H29*100`
- Row 39: `=(H16-H23)/H30*100`
- Row 40: `=(H17-H24)/H31*100`

### Step 2 continued – Statistics in H42:L47
For each column H through L:
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Use `PERCENTILE` (not `PERCENTILE.INC` or `_xlfn.PERCENTILE.INC`). When writing with openpyxl, the formula string must be exactly `=PERCENTILE(H35:H40,0.25)` — no `_xlfn.` prefix, no `.INC` suffix. Verify after writing that the cell's `.value` attribute contains exactly this string.

Also for MEDIAN: use `=MEDIAN(H35:H40)` — no `_xlfn.` prefix.

### Step 3 – Weighted mean in H50:L50
For each column H through L:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

### Save
Save to `/root/output/result.xlsx`. Do NOT use `data_only` mode. Keep all existing formatting.

## 3 – Verify
After saving, reopen the file with openpyxl and print the formula strings in cells H46, H47, H50 to confirm they contain the correct function names (PERCENTILE, SUMPRODUCT) without any `_xlfn.` prefix or `#NAME?` values.

## Important Notes
- Before writing formulas, inspect the actual data layout on the `Data` sheet to get the correct range references.
- The stat labels in rows 42–47 might be in a different order than MIN/MAX/MEDIAN/MEAN/P25/P75. Read the actual labels from the sheet and match accordingly.
- Do NOT add any new sheets, macros, VBA, or external links.
- Do NOT change any existing formatting.
- If openpyxl's formula writing adds `_xlfn.` prefixes automatically, work around it by setting cell values directly as strings starting with `=`.
- Double-check that MEDIAN and AVERAGE also don't get prefixed.

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