# Task Instruction

Execute the following steps to produce /root/output/result.xlsx.

## 0. Preparation
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl and print:
- Sheet names
- The contents of `Task!D12:D17`, `Task!D19:D24`, `Task!D26:D31` (series codes)
- The contents of `Task!H10:L10` (years)
- The contents of `Data!A21:A38` and `Data!B21:B38` (a few columns of the data source to understand its layout)
- The contents of `Task!B35:D40` (campus labels / series codes for the Net renewable balance block)
- The contents of `Task!B42:D47` (stat labels)
- The contents of `Task!B50:D50` (MCEC row)
- Any existing values/formulas already in H12:L17, H35:H40, H42:H47, H50

Print everything clearly so we understand the exact layout before writing formulas.

## 2. Write formulas with openpyxl (Python script)

After inspecting, write a Python script that:

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write a formula using INDEX/MATCH that:
- Looks up the series code from column D of the same row
- Looks up the year from row 10 of the same column
- Searches in `Data!$A$21:$A$38` for the series code (row match)
- Searches in `Data!$A$20:$XX$20` (or whatever the header row is) for the year (column match) — determine the exact header row and data range from the inspection step
- Pattern: `=INDEX(Data!<data_range>,MATCH(<series_code_cell>,Data!<series_code_column>,0),MATCH(<year_cell>,Data!<year_header_row>,0))`

Make sure the data range in INDEX covers all rows 21:38 and all relevant columns. Use absolute references (`$`) where needed for the lookup arrays.

### Step 2a – Net renewable balance in H35:L40
Based on the task description:
- `Net renewable balance = (Renewable Generation - Grid Consumption) / Baseline Energy Demand * 100`
- The Renewable Generation values are in H12:L17
- The Grid Consumption values are in H19:L24  
- The Baseline Energy Demand values are in H26:L31
- So for cell H35: `=(H12-H19)/H26*100`
- Apply this pattern across H35:L40

Verify from the inspection that the campus ordering in rows 12-17, 19-24, 26-31, and 35-40 all correspond (i.e., row 12 and row 19 and row 26 and row 35 are the same campus). If the ordering differs, adjust the cell references accordingly.

### Step 2b – Summary statistics in H42:L47
For each column H through L:
- Row 42 (min): `=MIN(H35:H40)`
- Row 43 (max): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE.INC(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE.INC(H35:H40,0.75)`

**CRITICAL**: For the percentile functions, openpyxl stores formulas as strings. The function name `PERCENTILE.INC` contains a dot. When writing with openpyxl, you MUST use `PERCENTILE.INC` exactly (not `PERCENTILE` which may not be recognized, and not `_xlfn.PERCENTILE.INC`). However, based on the previous failure with #NAME? errors, the issue is likely that openpyxl or the verifier needs the `_xlfn.` prefix for newer Excel functions. 

Try this approach: Write the formulas as `=_xlfn.PERCENTILE.INC(H35:H40,0.25)` — the `_xlfn.` prefix is how openpyxl internally represents modern Excel functions. If the inspection of the existing workbook reveals any existing formulas with `_xlfn.` prefix patterns, follow that convention.

Actually, the safest approach: use `PERCENTILE` (the legacy function without `.INC`) as the first attempt. If that doesn't work in the environment, the `_xlfn.` prefix version would be needed. Given the previous failure mentioned `#NAME?`, let's use **both a primary and fallback strategy**: First check if the test environment evaluates formulas or just pattern-matches. Use `PERCENTILE.INC` without any prefix — openpyxl should handle this correctly when you set `ws['H46'] = '=PERCENTILE.INC(H35:H40,0.25)'`.

Final decision on percentile syntax: Use `_xlfn.PERCENTILE.INC`. This is the standard way openpyxl represents this function internally and it will be recognized by Excel and most verifiers.

### Step 3 – Weighted mean in H50:L50
For each column H through L:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This computes the weighted mean using the Net renewable balance percentages as values and Baseline Energy Demand as weights.

## 3. Save and verify
- Save to `/root/output/result.xlsx`
- Reopen the saved file and print formulas in key cells (H12, H35, H42, H46, H47, H50) to confirm they were written correctly
- Check that no cells contain #NAME? or other error indicators in the formula strings
- Run the test suite: `cd /root && python -m pytest tests/ -v`

## Important constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs
- Do NOT modify existing formatting
- Work only inside sheets `Task` and `Data`
- Only modify the specified cell ranges

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