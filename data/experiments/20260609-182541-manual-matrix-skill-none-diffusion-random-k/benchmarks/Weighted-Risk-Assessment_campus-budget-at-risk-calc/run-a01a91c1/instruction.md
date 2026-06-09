# Task Instruction

Execute the following Python script to produce `/root/output/result.xlsx`. The script:

1. Opens `/root/data/workbook.xlsx` with openpyxl (data_only=False to preserve structure).
2. Inspects the `Data` sheet rows 21-38 to understand the layout (series codes column, year header row).
3. Inspects the `Task` sheet to understand the layout (column D series codes, row 10 years, yellow cell ranges).
4. Populates `H12:L17`, `H19:L24`, `H26:L31` with INDEX/MATCH formulas referencing the Data sheet.
5. Populates `H35:L40` with the Net budget buffer formula: `(Committed Funding - Operating Spend) / Approved Budget Base * 100`.
6. Populates `H42:L47` with MIN, MAX, MEDIAN, AVERAGE, 25th percentile, 75th percentile of the H35:L40 block (column-wise).
7. Populates `H50:L50` with SUMPRODUCT weighted mean formula.
8. Saves to `/root/output/result.xlsx`.

**CRITICAL**: For PERCENTILE.INC, use `_xlfn.PERCENTILE.INC` prefix. For MEDIAN, use `_xlfn.MEDIAN` if needed (test both). Same for any post-2007 functions.

**Detailed steps:**

```python
import openpyxl
import os

# Load workbook
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
ts = wb['Task']
ds = wb['Data']

# First, inspect the layout
print("=== Task sheet inspection ===")
print(f"Row 10 (years): {[ts.cell(row=10, column=c).value for c in range(1, 15)]}")
print(f"Row 11 (headers?): {[ts.cell(row=11, column=c).value for c in range(1, 15)]}")
for r in range(12, 52):
    row_data = [ts.cell(row=r, column=c).value for c in range(1, 15)]
    if any(v is not None for v in row_data):
        print(f"Row {r}: {row_data}")

print("\n=== Data sheet inspection ===")
# Check header row for years in Data sheet
for r in [1, 2, 3, 20, 21, 22]:
    print(f"Data Row {r}: {[ds.cell(row=r, column=c).value for c in range(1, 20)]}")
# Check a few data rows
for r in range(21, 39):
    print(f"Data Row {r}: {[ds.cell(row=r, column=c).value for c in range(1, 20)]}")
```

Run this inspection first. Then, based on the actual layout:

**Step 1 - INDEX/MATCH formulas:**
For each cell in H12:L17, H19:L24, H26:L31:
- The series code is in column D of that row on the Task sheet.
- The year is in row 10 of that column on the Task sheet.
- Use: `=INDEX(Data!$A$21:$Z$38, MATCH($D{row}, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$A$20:$Z$20, 0))`
- Adjust the Data ranges based on what the inspection reveals (find which column has series codes, which row has year headers, and the extent of the data).

**Step 2 - Net budget buffer (H35:L40):**
The three blocks are likely: Committed Funding (rows 12-17), Operating Spend (rows 19-24), Approved Budget Base (rows 26-31).
- Formula: `=(H12 - H19) / H26 * 100` (adjust row references for each department row).

**Step 3 - Statistics (H42:L47):**
For each column (H through L):
- H42: `=MIN(H35:H40)`
- H43: `=MAX(H35:H40)`
- H44: `=_xlfn.MEDIAN(H35:H40)` (try without prefix first if inspection shows MEDIAN works)
- H45: `=AVERAGE(H35:H40)`
- H46: `=_xlfn.PERCENTILE.INC(H35:H40,0.25)`
- H47: `=_xlfn.PERCENTILE.INC(H35:H40,0.75)`

Check the Task sheet labels in column A/B/C for rows 42-47 to determine the correct order of statistics.

**Step 4 - Weighted mean (H50:L50):**
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

After writing all formulas, save to `/root/output/result.xlsx`. Do NOT change any formatting. Do NOT add sheets.

Make sure to create the output directory if it doesn't exist: `os.makedirs('/root/output', exist_ok=True)`.

Run the inspection script first, then adapt the exact ranges and write the formulas accordingly.

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