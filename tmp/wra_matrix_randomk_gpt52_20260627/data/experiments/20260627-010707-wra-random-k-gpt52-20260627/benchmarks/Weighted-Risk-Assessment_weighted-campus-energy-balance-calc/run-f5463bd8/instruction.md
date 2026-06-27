# Task Instruction

Execute the following steps precisely to complete the weighted campus energy balance workbook.

## 0. Preparation

```bash
mkdir -p /root/output
pip install openpyxl
```

Then write and run a Python script that does the following:

## 1. Inspect the workbook

Open `/root/data/workbook.xlsx` with `openpyxl` (keep formatting: do NOT use `data_only=True`). Inspect:
- Sheet `Task`: print the values/formulas in columns A-G for rows 10-50 to understand the layout (series codes in column D, years in row 10, campus names, etc.).
- Sheet `Data`: print rows 21-38 fully (all columns) to understand the lookup source structure — identify which row contains headers, which column contains series codes, and how years are arranged.
- Print the exact contents of cells H10:L10 on `Task` (the year headers).
- Print column D for rows 12-31 on `Task` (the series codes for all three blocks).
- Print any existing content in rows 35-50 on `Task` (labels, existing formulas).

Do NOT proceed to editing until you have printed and understood all of this. Post the full output.

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write formulas using INDEX/MATCH (or VLOOKUP with MATCH, etc.) that:
- Look up the value from sheet `Data` rows 21:38
- Use two keys: the series code from column D of the current row on `Task`, and the year from row 10 of the current column on `Task`
- The formula pattern should be something like: `=INDEX(Data!$B$22:$Z$38,MATCH($D12,Data!$A$22:$A$38,0),MATCH(H$10,Data!$B$21:$Z$21,0))` — but adjust the exact ranges based on what you observe in the data layout.

IMPORTANT: Use `translate=False` when creating formulas in openpyxl, or simply assign the formula string directly to each cell. openpyxl stores formulas as strings starting with `=`.

For each cell in the three blocks (H12:L17, H19:L24, H26:L31), assign the appropriate formula string. Make sure:
- The series code reference uses an absolute column ($D) and relative row (e.g., $D12)
- The year reference uses a relative column and absolute row (e.g., H$10)
- The Data range references are fully absolute

## 3. Calculate Net Renewable Balance in H35:L40

The formula for each cell is:
`= (RenewableGeneration - GridConsumption) / BaselineEnergyDemand * 100`

where:
- Renewable Generation values are in H12:L17 (the first block)
- Grid Consumption values are in H19:L24 (the second block)  
- Baseline Energy Demand values are in H26:L31 (the third block)

So for cell H35: `=(H12-H19)/H26*100`
For cell H36: `=(H13-H20)/H27*100`
...and so on through L40.

Verify the row mapping by checking which campus appears in which row of each block. The six campuses in rows 35-40 should correspond to the same campuses in rows 12-17, 19-24, and 26-31 respectively.

## 4. Summary statistics in H42:L47

For each column H through L:
- H42: `=MIN(H35:H40)` (minimum)
- H43: `=MAX(H35:H40)` (maximum)
- H44: `=MEDIAN(H35:H40)` (median)
- H45: `=AVERAGE(H35:H40)` (simple mean)
- H46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- H47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

Check the row labels in column A/B/C for rows 42-47 to confirm the correct order of statistics. Adjust the row assignments if the labels indicate a different order.

## 5. Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This uses the Net Renewable Balance percentages as values and Baseline Energy Demand as weights.

## 6. Save and verify

Save to `/root/output/result.xlsx`. Then:
- Reopen the saved file and print all formula cells to verify they are stored correctly.
- Confirm no extra sheets were added.
- Confirm the file opens without errors.

## Critical constraints
- Do NOT delete or modify any existing content, formatting, or sheets.
- Do NOT add new sheets, macros, VBA, or external links.
- Do NOT use `data_only=True` when loading.
- When assigning formulas, ensure they start with `=`.
- Use `wb.save('/root/output/result.xlsx')` at the end.

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