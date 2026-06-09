# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

### 0 – Environment & Inspection
```
mkdir -p /root/output
pip install openpyxl
```
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and inspect:
- Sheet names (expect `Task` and `Data`).
- On `Task`: read row 10 (years in H10:L10), column D for rows 12-17, 19-24, 26-31 (series codes), and the layout of rows 35-50.
- On `Data`: read rows 21-38 to understand the lookup table structure (which row holds headers, which column holds series codes, where years appear).

Print all of this so you understand the exact layout before writing any formulas.

### 1 – Lookup formulas (H12:L17, H19:L24, H26:L31)
For each yellow cell in these three blocks, write an `INDEX/MATCH` formula string. The formula should:
- Use the series code from column D of the **current row** on `Task`.
- Use the year from row 10 of the **current column** on `Task`.
- Look up in `Data!$A$21:$A$38` (or wherever the series codes live) and `Data!` row that contains years (determine from inspection).
- Return the matching value from the data body on sheet `Data` rows 21-38.

Concrete pattern (adjust references after inspection):
```
=INDEX(Data!$B$21:$XX$38, MATCH(D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```
Adjust column letters and row numbers to match what you actually see. Use absolute references for the data range and mixed references for the lookup keys (anchor row for the year, anchor column for the series code).

### 2 – Net production slack (H35:L40)
Formula for each cell in H35:L40:
```
=(Hxx - Hyy) / Hzz * 100
```
where `xx` is the corresponding Finished Output row (12-17), `yy` is the Scrap And Rework row (19-24), and `zz` is the Rated Production Capacity row (26-31). Map plant 1→row 12/19/26 to row 35, plant 2→row 13/20/27 to row 36, etc.

### 3 – Summary statistics (H42:L47)
For each column (H through L):
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Use `PERCENTILE` (not `PERCENTILE.INC`). If you must use `PERCENTILE.INC`, prefix it with `_xlfn.` so openpyxl writes it correctly. Cross-task feedback shows that `PERCENTILE.INC` without the `_xlfn.` prefix causes `#NAME?` errors. Similarly, use `MEDIAN` not `_xlfn.MEDIAN`. Verify by reading back the cell values after writing.

### 4 – Weighted mean (H50:L50)
For each column col in H..L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
(Replace H with the appropriate column letter for each of the 5 columns.)

### 5 – Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets.

### 6 – Validation
Reload the saved file and:
1. Confirm all formula cells in the target ranges contain formula strings (start with `=`).
2. Confirm no cells are empty or contain plain values where formulas are expected.
3. Print a sample of formulas from each block for visual verification.
4. Specifically check rows 46-47 formulas do NOT contain unrecognized function names.

If any issue is found, fix and re-save before finishing.

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