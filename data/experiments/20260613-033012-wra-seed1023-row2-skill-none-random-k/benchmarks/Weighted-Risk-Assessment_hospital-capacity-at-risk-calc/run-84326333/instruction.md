# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 – Inspect the workbook
```python
import openpyxl, os, json
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for s in wb.sheetnames:
    print(s)
ws_task = wb['Task']
ws_data = wb['Data']

# Print Task sheet layout: columns A-L, rows 1-55
for r in range(1, 56):
    vals = []
    for c in range(1, 13):  # A=1 .. L=12
        cell = ws_task.cell(row=r, column=c)
        vals.append(f"{cell.value}")
    print(f"Row {r:2d}: {vals}")

print("\n--- Data sheet rows 18-40, cols A-Z ---")
for r in range(18, 41):
    vals = []
    for c in range(1, 27):
        cell = ws_data.cell(row=r, column=c)
        vals.append(f"{cell.value}")
    print(f"Row {r:2d}: {vals}")
```
Run this first to understand the exact layout: which column holds the series codes on Task (should be D), which row holds years (row 10), and how the Data sheet rows 21-38 are structured (orientation of lookup table, key columns/rows).

## 1 – Write the workbook with formulas

After inspecting, write a single Python script that:

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each yellow cell at row `r`, column `c` (H=8 … L=12):
- The series code is in column D of that row (`$D{r}`).
- The year is in row 10 of that column (`{col_letter}$10`).
- The lookup table is on sheet `Data`, rows 21:38.

Determine from inspection whether Data is arranged with series codes in a column (use VLOOKUP+MATCH or INDEX+MATCH) or in a row (use HLOOKUP+MATCH). Choose INDEX+MATCH as it is the most flexible:

```
=INDEX(Data!<data_range>, MATCH($D{r}, Data!<series_code_column>, 0), MATCH({col}$10, Data!<year_row>, 0))
```

Adjust the ranges based on what you see in the inspection. Lock references appropriately with `$`.

### Step 2 – Net capacity headroom in H35:L40
For each of the 6 hospital clusters (rows 35-40), the three input blocks are:
- Available Care Slots: rows 12-17 (offset = row - 23)
- Occupied Care Slots: rows 19-24 (offset = row - 16)
- Staffed Bed Capacity: rows 26-31 (offset = row - 9)

So for row 35, col H:
```
=(H12-H19)/H26*100
```
For row 36, col H:
```
=(H13-H20)/H27*100
```
And so on. Generalize: for row `r` in 35..40 and column `c` in H..L:
```
=({c}{r-23}-{c}{r-16})/{c}{r-9}*100
```

### Step 2 continued – Statistics in H42:L47
For each column `c` (H..L):
- Row 42 (MIN):    `=MIN({c}35:{c}40)`
- Row 43 (MAX):    `=MAX({c}35:{c}40)`
- Row 44 (MEDIAN): `=MEDIAN({c}35:{c}40)`
- Row 45 (MEAN):   `=AVERAGE({c}35:{c}40)`
- Row 46 (25th):   `=PERCENTILE({c}35:{c}40,0.25)`
- Row 47 (75th):   `=PERCENTILE({c}35:{c}40,0.75)`

**CRITICAL**: Use `PERCENTILE` — NOT `PERCENTILE.INC` or `PERCENTILE.EXC`. openpyxl does not support the dotted function names and they produce #NAME? errors. This was the exact failure mode in the campus-budget task iteration 0.

### Step 3 – Weighted mean in H50:L50
For each column `c`:
```
=SUMPRODUCT({c}35:{c}40,{c}26:{c}31)/SUM({c}26:{c}31)
```

### Save
```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## 2 – Validate
Reload the saved file and print cells in the formula regions to confirm formulas are present (not None) and are strings starting with `=`. Spot-check a few cells.

## Important constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Do NOT use `data_only=True` when loading.
- Adjust all ranges based on what you actually see in the inspection step. The row/column references above are guidance based on the task description; if the actual layout differs, adapt accordingly.
- After inspection, write all formulas in a single script and save once.

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