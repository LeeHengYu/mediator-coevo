# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

This task requires writing Excel formulas into specific cells of an existing workbook. Follow these steps precisely.

### Step 0: Setup and Inspect
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (without `data_only`) and inspect:
   - Sheet `Task`: Check what's in column D rows 12-17, 19-24, 26-31 (series codes). Check row 10 columns H-L (years). Check the existing structure of rows 35-50.
   - Sheet `Data`: Check rows 21-38 to understand the data layout. Specifically note row 21 (header row with years) and column A rows 22-38 (series codes). Note the exact column range of the data.
3. Print all these values so you can construct correct formulas.

### Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31
For each cell in these ranges, write an INDEX-MATCH formula:
```
=INDEX(Data!$B$22:$Z$38, MATCH($D{row}, Data!$A$22:$A$38, 0), MATCH({col}$10, Data!$B$21:$Z$21, 0))
```
Where `{row}` is the current row number and `{col}` is the current column letter (H, I, J, K, L).

**Important**: Before writing formulas, verify the exact data range on the Data sheet. The column range for data values and the row range for series codes must match exactly. Adjust `$B$22:$Z$38`, `$A$22:$A$38`, and `$B$21:$Z$21` if the actual data layout differs.

### Step 2: Net Patient Flow in H35:L40
For each cell in H35:L40, calculate:
```
=(H12 - H19) / H26 * 100
```
Adjusted for the correct row offsets:
- Row 35 uses data from rows 12, 19, 26 (first hospital)
- Row 36 uses data from rows 13, 20, 27 (second hospital)
- Row 37 uses data from rows 14, 21, 28 (third hospital)
- Row 38 uses data from rows 15, 22, 29 (fourth hospital)
- Row 39 uses data from rows 16, 23, 30 (fifth hospital)
- Row 40 uses data from rows 17, 24, 31 (sixth hospital)

So for cell H35: `=(H12-H19)/H26*100`, for H36: `=(H13-H20)/H27*100`, etc.

### Step 3: Statistics in H42:L47
For each column (H through L):
- Row 42 (Min): `=MIN(H35:H40)`
- Row 43 (Max): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Use `PERCENTILE` (legacy), NOT `PERCENTILE.INC`. The latter causes `#NAME?` errors in the evaluation environment.

### Step 4: Weighted Mean in H50:L50
For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

### Step 5: Save the workbook
Save to `/root/output/result.xlsx` using openpyxl.

### Step 6: Force recalculation with LibreOffice
Run:
```bash
libreoffice --headless --calc --convert-to xlsx --outdir /root/output /root/output/result.xlsx
```
This forces all formulas to be evaluated and cached values to be written, which is required by the verifier that reads with `data_only=True`.

### Step 7: Verify
Open `/root/output/result.xlsx` with openpyxl using `data_only=True` and spot-check:
- A few lookup cells (e.g., H12, L31) have numeric values (not None, not #NAME?)
- A few net flow cells (e.g., H35) have numeric values
- Statistics cells H42:L47 all have numeric values (especially H46, H47 - the percentiles)
- Weighted mean cells H50:L50 have numeric values

If any cell shows None or an error string, debug before finishing.

### Important Constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Do NOT use `PERCENTILE.INC` or `PERCENTILE.EXC` - use only `PERCENTILE`.
- Use `MEDIAN` not `MEDIAN.INC`.
- Use `AVERAGE` for the simple mean.

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