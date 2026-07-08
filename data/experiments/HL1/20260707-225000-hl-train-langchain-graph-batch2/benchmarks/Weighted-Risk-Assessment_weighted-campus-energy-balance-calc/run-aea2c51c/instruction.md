# Task Instruction

Execute the following steps carefully to produce `/root/output/result.xlsx`.

## 0. Preparation

```bash
mkdir -p /root/output
pip install openpyxl
```

Open and inspect `/root/data/workbook.xlsx` with openpyxl to understand the layout:
- Sheet `Task`: look at column D rows 12-17, 19-24, 26-31 for series codes; row 10 columns H-L for years; rows 35-40, 42-47, 50 for where results go.
- Sheet `Data`: look at rows 21-38 to understand the data table structure (which row/column holds series codes, which holds years, where values start).

Print all of this to stdout so you can build correct formulas.

## 1. Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each yellow cell, write a formula that looks up the value from `Data!` rows 21:38 using:
- The series code from column D of the same row on `Task`
- The year from row 10 of the same column on `Task`

Use `INDEX(MATCH,MATCH)` pattern. Before writing formulas, inspect the Data sheet to determine:
- Whether series codes are in a column (for MATCH on rows) or a row (for MATCH on columns)
- Whether years are in a row (for MATCH on columns) or a column (for MATCH on rows)

Construct the INDEX/MATCH formula accordingly. Example pattern (adjust references after inspection):
`=INDEX(Data!$B$22:$Z$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$Z$21, 0))`

Adjust the exact ranges based on what you find in the Data sheet.

## 2. Net renewable balance in H35:L40

Formula for each cell: `=(H12 - H19) / H26 * 100` (adjusting row references for each campus row, i.e., row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.)

## 3. Statistics in H42:L47

For each column (H through L):
- Row 42 (Min): `=MIN(H35:H40)`
- Row 43 (Max): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: For the percentile functions, try BOTH `PERCENTILE` and `PERCENTILE.INC` to see which one the openpyxl/evaluation environment accepts. The previous run failed with #NAME? errors on these cells. To be safe:
- First, write the formulas using `PERCENTILE` (without `.INC` or `.EXC` suffix), as this is the most universally recognized form.
- After saving, open the file with openpyxl in read-only/data-only mode or run the test to check if the formulas evaluate correctly.
- If `PERCENTILE` causes #NAME?, switch to `PERCENTILE.INC`.
- If `PERCENTILE.INC` also fails, try `_xlfn.PERCENTILE.INC` (the internal Excel prefix openpyxl sometimes needs).

The safest approach: write `=PERCENTILE(H35:H40,0.25)` first. If the verifier rejects it, then use `=_xlfn.PERCENTILE.INC(H35:H40,0.25)`. Try the plain name first.

## 4. Weighted mean in H50:L50

For each column: `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

## 5. Save

Save to `/root/output/result.xlsx`. Do NOT add new sheets, macros, VBA, or external links. Preserve existing formatting.

## 6. Validate

After saving, re-open the file and:
1. Print the formulas in cells H46 and H47 to confirm they don't contain unrecognized function names.
2. Run the test if available: `cd /root && python -m pytest tests/ -v` or similar.
3. If the test shows #NAME? errors on percentile cells, re-edit using the `_xlfn.PERCENTILE.INC` prefix and re-save, then re-test.

## Important Notes
- Use openpyxl to write formulas (not values).
- When writing formulas with openpyxl, the formula string should start with `=`.
- Do NOT use `data_only=True` when loading for editing; that strips formulas.
- Load with `keep_vba=False` (default) and do not alter formatting.

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