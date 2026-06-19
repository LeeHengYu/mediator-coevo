# Task Instruction

You need to update `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Work only inside the existing sheets `Task` and `Data`; do not add sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

**Before writing any formulas, inspect the workbook carefully:**
1. Open `/root/data/workbook.xlsx` with openpyxl and inspect both sheets.
2. On sheet `Task`: read row 10 to find the year headers in columns H–L. Read column D in the relevant row ranges (12–17, 19–24, 26–31) to find the series codes. Note the exact text of each series code and each year value.
3. On sheet `Data`: read rows 21–38 to understand the data layout — identify which column holds the series codes and which row/column holds the year headers. Determine the orientation (whether years are in a row header and series codes in a column, or vice versa).
4. Print out these inspected values so you can construct correct formulas.

**Step 1 — Lookup formulas in H12:L17, H19:L24, H26:L31:**
For each yellow cell in these three blocks, write a spreadsheet formula using `INDEX` with `MATCH`. Each formula should look up the value from `Data!` rows 21:38 using two keys: the series code from column D of the current row on `Task`, and the year from row 10 on `Task`. Use absolute references for the data range on `Data` and mixed references so the formula can be understood per-cell. Make sure the MATCH functions reference the correct row/column for series codes and the correct row/column for years on the Data sheet.

Example pattern (adjust ranges based on your inspection):
```
=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```
Adjust `$A$21:$A$38` to the actual column containing series codes, `$B$20:$XX$20` to the actual row containing year headers, and `$B$21:$XX$38` to the actual data range — all based on your inspection.

**Step 2 — Net reliability gap in H35:L40:**
For each of the six regions (rows 35–40) and each year column (H–L), calculate:
```
= (Successful_API_Requests - Failed_API_Requests) / Compute_Capacity * 100
```
where:
- `Successful API Requests` values are in H12:L17
- `Failed API Requests` values are in H19:L24
- `Compute Capacity` values are in H26:L31

The regions in rows 35–40 must correspond to the same regions in the three blocks above. Inspect column D (or equivalent label column) for rows 35–40 and rows 12–17 to confirm the region ordering matches. If the ordering differs, map each row in 35–40 to the correct row in 12–17, 19–24, 26–31 by matching region names.

Write cell formulas like: `=(H12-H19)/H26*100` (adjusting row references per region).

**Step 2 continued — Summary statistics in H42:L47:**
For each year column H–L, calculate these over the six Net reliability gap values (H35:H40 etc.):
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

Verify the row assignments by reading the labels in column B or D for rows 42–47. Assign formulas to match the correct label (min, max, median, mean, 25th percentile, 75th percentile).

**Step 3 — Weighted mean in H50:L50:**
For each year column, compute:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the Net reliability gap percentages as values and Compute Capacity as weights.

**Final steps:**
1. After writing all formulas, save the workbook to `/root/output/result.xlsx` (create `/root/output/` if needed).
2. Re-open the saved file and verify that cells H12, H35, H42, and H50 contain formulas (not None), and that when evaluated they produce numeric results. Print several cell values to confirm.
3. If any cell is None or empty, debug by re-inspecting the Data sheet layout and fix the formulas.

**Critical warnings:**
- Do NOT leave any target cells empty. A previous failed run on a similar task returned None for all cells because formulas were never written.
- Use `data_only=False` when writing formulas (the default). Do not set `data_only=True`.
- When saving with openpyxl, formulas will be stored as strings starting with `=`. They will be evaluated when opened in Excel or by the verifier.
- Make sure to preserve existing cell formatting by not overwriting style attributes — only set the `.value` property of each cell.

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