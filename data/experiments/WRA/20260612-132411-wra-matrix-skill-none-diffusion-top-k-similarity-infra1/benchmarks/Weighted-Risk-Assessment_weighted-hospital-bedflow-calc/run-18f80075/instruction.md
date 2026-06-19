# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 – Preparation
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1 – Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Print:
- Sheet names.
- The contents of `Task!D12:D17`, `Task!D19:D24`, `Task!D26:D31` (series codes).
- The contents of `Task!H10:L10` (year headers).
- The contents of `Data!A21:A38` and `Data!B21:B38` (first two columns of the lookup source) so we understand the data layout.
- The contents of `Data!1:1` or the first row of the data block to understand column headers.
- The contents of `Task!H35:H40` area labels and `Task!D35:D40`.
- The contents of `Task!H42:H47` area labels and `Task!D42:D47`.
- The contents of `Task!H50:L50` and `Task!D50`.

Print everything verbatim so we can craft exact formulas.

## 2 – Write formulas (Python / openpyxl)
After inspecting, write a Python script that:

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three 6×5 blocks, write an `INDEX/MATCH` formula that:
- Finds the row in `Data!$A$21:$A$38` matching the series code in column D of the current Task row (e.g., `Task!$D12`).
- Finds the column in `Data!$21:$21` (or whichever row holds the year headers on the Data sheet) matching the year in `Task!H$10` (adjusting the column letter for each column H–L).
- Returns the value from `Data!$A$21:$Z$38` (or the appropriate rectangular range).

Use the inspected layout to set exact range references. The formula pattern per cell (example for H12):
```
=INDEX(Data!$B$21:$XX$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$XX$20,0))
```
Adjust the column extent based on what you find during inspection. The key is that the MATCH for the year scans the header row of the Data block, and the MATCH for the series code scans the first column.

### Step 2 – Net patient flow in H35:L40
For each cell in H35:L40, write a formula:
```
=(H12-H19)/H26*100
```
(adjusting row references: row 12→admissions block row, row 19→discharges block row, row 26→capacity block row, for the corresponding hospital in the same relative position within each block).

Specifically:
- H35 = (H12-H19)/H26*100
- H36 = (H13-H20)/H27*100
- ... through H40 = (H17-H24)/H31*100
And similarly for columns I, J, K, L.

### Step 2 continued – Summary stats in H42:L47
For each column (H through L):
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th pctl): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th pctl): `=PERCENTILE(H35:H40,0.75)`

Check the labels in column D (or nearby) during inspection to confirm the exact order of min/max/median/mean/25th/75th. Adjust row assignments to match whatever labels are already present.

### Step 3 – Weighted mean in H50:L50
For each column (H through L):
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```

## 3 – Save
Save the workbook to `/root/output/result.xlsx`. Do NOT use `data_only=True` when loading (that strips formulas). Keep all existing formatting.

## 4 – Validate
After saving, reopen `/root/output/result.xlsx` with openpyxl (data_only=False) and print the formula content of cells H12, L17, H35, L40, H42, L47, H50, L50 to confirm they are non-empty formula strings (starting with '=').

Also check that no existing formatting or sheets were altered by comparing sheet names and spot-checking a few formatted cells.

## IMPORTANT NOTES
- Assign formula strings (starting with `=`) to cell `.value` attributes. Do NOT try to compute numeric values in Python.
- Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
- Do NOT use `data_only=True` when loading.
- The inspection step is critical: print the actual cell contents before writing any formulas so the range references are correct.
- If the Data sheet header row for years is not row 20, adjust all formulas accordingly based on what you find.
- Make sure MATCH lookups use exact match (0 as the third argument).
- Use absolute row references for the year header row (`$10`) and absolute column references for the series code column (`$D`) so formulas can be placed across the block without breaking.

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