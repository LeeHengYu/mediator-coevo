# Task Instruction

Reproduce the successful prior run. Concrete steps:

1. **Copy the workbook** so you always have the original:
 ```bash
 cp /root/data/workbook.xlsx /root/output/result.xlsx
 ```

2. **Inspect the workbook** with openpyxl (data_only=False) to collect:
 - Sheet `Task`: the series codes in column D for rows 12-17, 19-24, 26-31.
 - Sheet `Task`: the years in row 10 for columns H-L.
 - Sheet `Data`: confirm the data layout in rows 21-38 (which column holds the series code key, and how years are arranged across columns).
 - Sheet `Task`: the port names / order in rows 35-40 and the weights block H26:L31.
 - Any existing content or formatting you must preserve.

3. **Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31**

 For every cell (r, c) in these three blocks, write an INDEX/MATCH formula that:
 - Looks up the series code from cell D{r} in the key column of sheet `Data` rows 21-38.
 - Looks up the year from cell {col_letter}10 in the header row of sheet `Data`.
 - Returns the intersection value.

 Use the pattern:
 ```
 =INDEX(Data!<data_range>, MATCH($D{r}, Data!<key_column_range>, 0), MATCH({col}10, Data!<year_header_range>, 0))
 ```
 Adjust `<data_range>`, `<key_column_range>`, and `<year_header_range>` based on what you observe in step 2. Make sure all range references are correct and use the exact sheet name `Data`.

4. **Step 2 – Net container flow (H35:L40)**

 The three blocks from Step 1 correspond to three metrics. Identify which block is Loaded Containers Inbound (rows 12-17), which is Loaded Containers Outbound (rows 19-24), and which is Terminal Throughput Capacity (rows 26-31) by reading the labels in the Task sheet.

 For each cell in H35:L40, write:
 ```
 =({inbound_cell} - {outbound_cell}) / {capacity_cell} * 100
 ```
 where `{inbound_cell}`, `{outbound_cell}`, `{capacity_cell}` are the corresponding cells from the three blocks (same column, matching port row).

5. **Step 2 – Summary statistics (H42:L47)**

 For each column col in H through L, write these six formulas in rows 42-47:
 - Row 42 (MIN):  `=MIN({col}35:{col}40)`
 - Row 43 (MAX):  `=MAX({col}35:{col}40)`
 - Row 44 (MEDIAN): `=MEDIAN({col}35:{col}40)`
 - Row 45 (MEAN): `=AVERAGE({col}35:{col}40)`
 - Row 46 (25th pct): `=PERCENTILE({col}35:{col}40, 0.25)`
 - Row 47 (75th pct): `=PERCENTILE({col}35:{col}40, 0.75)`

 **CRITICAL**: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`). The avoid-artifact from the cloud-reliability task shows that `#NAME?` errors in the percentile rows caused failure — likely from using a function name the verifier doesn't recognize. Stick to the classic `PERCENTILE` function.

 Similarly use `AVERAGE` not `MEAN`, and `MEDIAN` not any variant.

6. **Step 3 – Weighted mean (H50:L50)**

 For each column col in H through L:
 ```
 =SUMPRODUCT({col}35:{col}40, {col}26:{col}31) / SUM({col}26:{col}31)
 ```
 This computes the weighted mean of the net-container-flow percentages using Terminal Throughput Capacity as weights.

7. **Verify row labels** — Before writing formulas, read the Task sheet labels for rows 42-47 to confirm which row is MIN, MAX, MEDIAN, MEAN, 25th percentile, 75th percentile. Adjust the row assignments if they differ from my assumption above.

8. **Save** the workbook (openpyxl wb.save) to `/root/output/result.xlsx`. Do NOT use data_only mode when loading.

9. **Post-save verification**: Re-open the saved file with openpyxl (data_only=False) and print the formula content of a sample of cells (e.g., H12, H35, H42, H46, H50) to confirm formulas were written correctly and no cells are empty or contain literal values where formulas are expected.

10. If a test script exists at `/root/tests/test_outputs.py` or similar, run it with `pytest` and report the result.

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