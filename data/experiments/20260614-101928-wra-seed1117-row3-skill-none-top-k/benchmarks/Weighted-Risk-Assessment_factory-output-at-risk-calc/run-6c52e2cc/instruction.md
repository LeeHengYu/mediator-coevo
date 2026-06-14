# Task Instruction

Execute the following steps in order to produce `/root/output/result.xlsx`.

## 0 – Inspect the workbook
```bash
mkdir -p /root/output
```
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and inspect:
- Sheet `Task`: read the series codes in column D for rows 12-17, 19-24, 26-31 and the years in H10:L10. Note exact text values.
- Sheet `Data`: read the header row and rows 21-38 to understand the layout (which column holds the series code, which row holds which year, etc.).
- Note the exact column letters/numbers so formulas reference the right cells.

Print all of this information before writing any formulas.

## 1 – Write lookup formulas in the yellow cells (H12:L17, H19:L24, H26:L31)

Use `INDEX/MATCH` pattern. For each yellow cell at row `r`, column `c` (H–L):

```
=INDEX(Data!$B$21:$<lastcol>$38, MATCH($D<r>, Data!$A$21:$A$38, 0), MATCH(<year_cell>, Data!$B$20:$<lastcol>$20, 0))
```

Adapt the exact range references based on what you observed in step 0. The series code anchor is column D of the current row on sheet Task; the year anchor is the cell in row 10 of the current column on sheet Task.

IMPORTANT: Use absolute references for the Data range and relative/mixed references for the lookup values so the formula can be written per-cell correctly.

## 2 – Net production slack (H35:L40)

For each plant row (6 plants, rows 35-40) and each year column (H-L):

```
=(H12 - H19) / H26 * 100
```

where H12 corresponds to the Finished Output block (rows 12-17), H19 to Scrap And Rework (rows 19-24), and H26 to Rated Production Capacity (rows 26-31). The row offsets must align: row 35 uses rows 12, 19, 26; row 36 uses 13, 20, 27; etc.

## 3 – Summary statistics (H42:L47)

For each year column c in H–L, write these formulas referencing the Net production slack block H35:L40 (same column):
- Row 42 (Min):    `=MIN(H35:H40)`
- Row 43 (Max):    `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean):   `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)`

⚠️ CRITICAL: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`). The failed artifact from campus-budget-at-risk-calc shows that using unrecognized function names like PERCENTILE.INC caused #NAME? errors. Stick to `PERCENTILE` which is universally recognized.

## 4 – Weighted mean for Regional Output Council (H50:L50)

For each year column c:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net production slack percentages using Rated Production Capacity as weights.

## 5 – Save and verify

Save the workbook to `/root/output/result.xlsx`.

After saving, reopen the file with openpyxl (data_only=False) and print the formulas in a sample of cells (e.g., H12, L17, H35, L40, H42, H47, H50, L50) to confirm they are correctly written as formula strings (not values, not None).

Also verify:
- No new sheets were added
- Only the `Task` and `Data` sheets exist
- The formulas look syntactically correct

## Key constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change any existing formatting.
- Use openpyxl to read and write. When writing formulas, assign string formulas (starting with `=`) to cells.
- Check the exact row/column layout from the inspection step before hardcoding any references. The row labels in the Task sheet and the Data sheet structure determine everything.

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