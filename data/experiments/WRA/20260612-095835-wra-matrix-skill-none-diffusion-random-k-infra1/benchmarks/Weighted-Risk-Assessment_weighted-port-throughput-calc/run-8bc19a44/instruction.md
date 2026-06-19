# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx` from `/root/data/workbook.xlsx`.

## Step 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and print:
   - Sheet names.
   - `Task` sheet: values in D12:D17, D19:D24, D26:D31 (series codes), H10:L10 (years), labels in A35:G40, A42:G47, A50:G50.
   - `Data` sheet: row 20 (header row with years), column B rows 21–38 (series codes), and a sample of data values (e.g., C21:G21) to confirm the layout.
   - Check whether cells H12:L17 etc. already contain formulas or are empty.

This inspection is critical — do NOT skip it. The exact row/column layout of the Data sheet (which column holds series codes, which row holds years, how many columns of year data) determines every formula.

## Step 1 – Write lookup formulas in H12:L17, H19:L24, H26:L31
Using openpyxl, write INDEX/MATCH formulas into each cell. The pattern for cell (r, c) should be:

```
=INDEX(Data!$C$21:$G$38, MATCH($D{r}, Data!$B$21:$B$38, 0), MATCH({col}$10, Data!$C$20:$G$20, 0))
```

where `{r}` is the Task-sheet row number and `{col}` is the column letter (H–L). Use `$D{r}` (mixed: column absolute, row relative within block) and `{col}$10` (column relative, row absolute) so the formula is correct per-cell.

**Important**: Adjust the Data-sheet ranges based on what you observed in Step 0. If years are in a different row or series codes in a different column, adapt accordingly. The failed run (weighted-hospital-bedflow-calc) wrote None because formulas targeted wrong ranges.

## Step 2 – Net container flow (H35:L40)
For each cell in H35:L40, write a formula that computes:
```
=(H12 - H19) / H26 * 100
```
with appropriate row offsets: row 35 uses rows 12, 19, 26; row 36 uses 13, 20, 27; etc.

Use cell references like `={col}{r_loaded_in} - {col}{r_loaded_out}) / {col}{r_capacity} * 100` where the three blocks are offset by the same index (0–5).

## Step 3 – Summary statistics (H42:L47)
For each column H–L, write these formulas in the six rows 42–47. Check the labels in column A/B to confirm which row is which statistic. The expected mapping (verify against actual labels) is:
- Row 42: `=MIN({col}35:{col}40)`
- Row 43: `=MAX({col}35:{col}40)`
- Row 44: `=MEDIAN({col}35:{col}40)`
- Row 45: `=AVERAGE({col}35:{col}40)`
- Row 46: `=PERCENTILE({col}35:{col}40, 0.25)`
- Row 47: `=PERCENTILE({col}35:{col}40, 0.75)`

**Critically**: Read the actual row labels before assigning formulas. If Min is in row 43 instead of 42, adjust accordingly.

## Step 4 – Weighted mean (H50:L50)
For each column H–L:
```
=SUMPRODUCT({col}35:{col}40, {col}26:{col}31) / SUM({col}26:{col}31)
```

## Step 5 – Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, add sheets, macros, or VBA.

## Step 6 – Verify
1. Re-open `/root/output/result.xlsx` with openpyxl and print the formula content of representative cells: H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50.
2. Confirm none are None/empty.
3. If a test script exists at `/root/test_output.py` or similar, run `cd /root && python -m pytest test_output.py -v` and report results.

## Key Pitfalls to Avoid
- Do NOT assume the Data sheet layout — inspect it first.
- Do NOT leave any target cell without a formula (the failed run wrote None).
- Do NOT use data_only=True when writing formulas.
- Do NOT add or remove sheets.
- Verify formulas are strings starting with '=' when read back.

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