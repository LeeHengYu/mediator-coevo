# Task Instruction

## Task: Update /root/data/workbook.xlsx with formulas and save to /root/output/result.xlsx

### Preparation
1. `mkdir -p /root/output`
2. Inspect the workbook structure first:
   - Open `/root/data/workbook.xlsx` using openpyxl.
   - Read sheet `Data` rows 21-38 to understand the data layout (columns, series codes, years).
   - Read sheet `Task` to understand the layout: column D series codes, row 10 years, yellow cell regions, and existing content in rows 35-50.
   - Print column D values for rows 12-17, 19-24, 26-31 (series codes).
   - Print row 10 values for columns H-L (years).
   - Print row 35 label area and rows 35-50 column D/G labels to understand the Net reliability gap and statistics layout.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a formula that looks up data from sheet `Data` rows 21:38. Use the INDEX/MATCH pattern:

```
=INDEX(Data!$B$21:$Z$38, MATCH($D{row}, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```

**IMPORTANT**: Before writing formulas, inspect the Data sheet to determine:
- Which column contains the series codes (likely column A on Data sheet)
- Which row contains the year headers (likely row 20 on Data sheet)
- The exact data range boundaries (first data column through last data column)

Adjust the INDEX/MATCH references accordingly based on what you find. The key is:
- Row lookup: MATCH the series code from column D of the Task sheet against the series code column in Data rows 21:38
- Column lookup: MATCH the year from row 10 of the Task sheet against the year header row in Data sheet

Write these formulas for all cells in the three blocks (rows 12-17, 19-24, 26-31, columns H-L).

### Step 2: Net reliability gap in H35:L40 and statistics in H42:L47

**Net reliability gap (H35:L40)**:
For each of the 6 regions (rows 35-40), calculate:
```
=(H{api_success_row} - H{api_fail_row}) / H{compute_row} * 100
```
where:
- `api_success_row` corresponds to the matching region in the H12:L17 block (Successful API Requests)
- `api_fail_row` corresponds to the matching region in the H19:L24 block (Failed API Requests)  
- `compute_row` corresponds to the matching region in the H26:L31 block (Compute Capacity)

So for row 35: `=(H12-H19)/H26*100`, row 36: `=(H13-H20)/H27*100`, etc. through row 40.

**Statistics (H42:L47)**:
- H42 (Min): `=MIN(H35:H40)`
- H43 (Max): `=MAX(H35:H40)`
- H44 (Median): `=MEDIAN(H35:H40)`
- H45 (Mean): `=AVERAGE(H35:H40)`
- H46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- H47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Use `PERCENTILE` (legacy function), NOT `PERCENTILE.INC`. The previous execution failed because the function name was not recognized. When writing with openpyxl, the formula string must be exactly `=PERCENTILE(H35:H40,0.25)` — no dots in the function name. Verify after writing that the stored formula string contains `PERCENTILE(` and not `PERCENTILE.INC(`.

Repeat for columns H through L.

### Step 3: Weighted mean in H50:L50

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net reliability gap percentages using Compute Capacity as weights.

### Final Steps
1. Save the workbook to `/root/output/result.xlsx`.
2. **Verification**: Re-open the saved file and:
   - Print the formula stored in cell H46 to confirm it says `=PERCENTILE(H35:H40,0.25)` (no #NAME?, no PERCENTILE.INC).
   - Print the formula stored in cell H47 to confirm it says `=PERCENTILE(H35:H40,0.75)`.
   - Print formulas from a sample lookup cell (e.g., H12) to verify the INDEX/MATCH pattern.
   - Print the formula from H35 to verify the net reliability gap calculation.
   - Print the formula from H50 to verify the SUMPRODUCT weighted mean.
   - Confirm no new sheets were added.
   - Confirm the file exists at `/root/output/result.xlsx`.

### Constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Use openpyxl for all operations.
- When writing formulas, ensure they start with `=`.

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