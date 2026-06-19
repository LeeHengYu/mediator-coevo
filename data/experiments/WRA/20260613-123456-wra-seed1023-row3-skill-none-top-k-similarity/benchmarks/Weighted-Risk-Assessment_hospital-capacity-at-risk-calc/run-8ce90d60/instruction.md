# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 – Inspect the workbook
```bash
mkdir -p /root/output
```
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Inspect:
- Sheet `Task`: read row 10 (years in H10:L10), column D for rows 12-17, 19-24, 26-31 (series codes), row labels in column B or C for rows 35-47, and any existing content/formatting in the target ranges.
- Sheet `Data`: read row 20 (headers), column A rows 21-38 (series codes), and a sample of data cells to confirm layout (headers in row 20, codes in column A, yearly data in columns B-F or similar).

Print all of these so you can build correct formulas.

## 1 – Populate H12:L31 with INDEX/MATCH formulas

For each cell in ranges H12:L17, H19:L24, H26:L31, write an Excel formula of the form:
```
=INDEX(Data!$B$21:$F$38, MATCH($D{row}, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$F$20, 0))
```
Adjust the data columns (B:F vs others) based on what you found in step 0. The key requirements:
- The series code reference must lock the column ($D) but keep the row relative ({row}).
- The year reference must lock the row ($10) but keep the column relative (H, I, J, K, L).
- The data range references on sheet `Data` must be fully absolute.

Verify by reading back a few cells that they contain formula strings (not values).

## 2 – Net capacity headroom in H35:L40

For each of the 6 hospital clusters (rows 35-40 corresponding to clusters in rows 12-17 / 19-24 / 26-31):
```
=(H{available_row} - H{occupied_row}) / H{staffed_row} * 100
```
where:
- `available_row` = rows 12-17 (Available Care Slots block)
- `occupied_row` = rows 19-24 (Occupied Care Slots block)
- `staffed_row` = rows 26-31 (Staffed Bed Capacity block)

Map cluster index 0-5 so row 35 uses rows 12, 19, 26; row 36 uses 13, 20, 27; etc.

## 3 – Summary statistics in H42:L47

Based on the row labels already present in column B/C for rows 42-47, write column-wise formulas over H35:H40 (through L35:L40):
- Minimum → `=MIN(H35:H40)`
- Maximum → `=MAX(H35:H40)`
- Median → `=MEDIAN(H35:H40)`
- Mean → `=AVERAGE(H35:H40)`
- 25th percentile → `=PERCENTILE(H35:H40,0.25)`   (**use PERCENTILE, not PERCENTILE.INC or PERCENTILE.EXC** — the avoid artifact warns that #NAME? errors came from using unsupported function names)
- 75th percentile → `=PERCENTILE(H35:H40,0.75)`

IMPORTANT: Read the actual row labels to confirm which row gets which function. If labels say something like P25/P75 or 25th/75th percentile, map accordingly. Use `PERCENTILE` (not `PERCENTILE.INC`) to avoid #NAME? errors in openpyxl/Excel compatibility.

## 4 – Weighted mean in H50:L50

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net capacity headroom percentages using Staffed Bed Capacity as weights.

## 5 – Save and verify

Save to `/root/output/result.xlsx`. Then reopen with openpyxl (data_only=False) and:
- Print formulas in a sample of cells from each range (H12, L17, H19, L24, H26, L31, H35, L40, H42, L47, H50, L50).
- Confirm no cells are empty or contain plain values where formulas are expected.
- Confirm PERCENTILE is used (not PERCENTILE.INC or PERCENTILE.EXC).
- Confirm no new sheets were added.

Do NOT add sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

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