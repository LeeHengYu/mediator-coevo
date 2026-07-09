# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx` from `/root/data/workbook.xlsx`.

## Step 0 – Setup & Inspection
```bash
mkdir -p /root/output
pip install openpyxl
```
Open `/root/data/workbook.xlsx` with openpyxl (keep `data_only=False` so formulas are preserved). Inspect:
- Sheet `Task`: confirm column D has series codes starting at rows 12-17, 19-24, 26-31. Confirm row 10 has year headers in columns H-L. Confirm the yellow target ranges and the layout described below.
- Sheet `Data`: confirm rows 21-38 contain the source data, and identify which column holds the series codes and which row holds the years (needed for MATCH orientation).

Print the contents of Task!D12:D31, Task!H10:L10, and Data rows 21-38 (first few columns and header row) so you understand the exact layout before writing any formulas.

## Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31

For each of the three blocks (rows 12-17, 19-24, 26-31), and for each cell in columns H through L, write a formula using the `INDEX/MATCH` pattern with mixed references so it can conceptually fill across columns and down rows:

```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```

**Adjust the exact ranges** after inspecting the Data sheet layout:
- The first MATCH locks column D with `$D` and lets the row float (e.g., `$D12`, `$D13`, …).
- The second MATCH locks the row with `$10` and lets the column float (e.g., `H$10`, `I$10`, …).
- The INDEX data range and MATCH lookup vectors must cover exactly the data area on sheet Data rows 21-38.
- Verify by reading back a few cells that the formulas are correctly placed and references shift per row/column.

## Step 2 – Net reliability gap (H35:L40) and summary statistics (H42:L47)

### H35:L40 – Net reliability gap per region
For each region row i (0-5 corresponding to the six regions):
```
H35+i = (H12+i - H19+i) / H26+i * 100
```
Concretely:
- H35 = `=(H12-H19)/H26*100`
- H36 = `=(H13-H20)/H27*100`
- H37 = `=(H14-H21)/H28*100`
- H38 = `=(H15-H22)/H29*100`
- H39 = `=(H16-H23)/H30*100`
- H40 = `=(H17-H24)/H31*100`

Repeat the same pattern for columns I, J, K, L (adjusting column letter, keeping the row offsets). **Critical**: each row must reference its own corresponding rows in the three blocks above – do NOT use a single absolute row for all six regions.

### H42:L47 – Summary statistics (column-wise over H35:H40)
For each column c in {H, I, J, K, L}:
- Row 42 (MIN):    `=MIN(c35:c40)`
- Row 43 (MAX):    `=MAX(c35:c40)`
- Row 44 (MEDIAN): `=MEDIAN(c35:c40)`
- Row 45 (MEAN):   `=AVERAGE(c35:c40)`
- Row 46 (25th %): `=PERCENTILE(c35:c40, 0.25)`
- Row 47 (75th %): `=PERCENTILE(c35:c40, 0.75)`

Check the labels in column D or G of rows 42-47 to confirm the correct order of min/max/median/mean/25th/75th and adjust row assignments if needed.

## Step 3 – Weighted mean in H50:L50
For each column c in {H, I, J, K, L}:
```
=SUMPRODUCT(c35:c40, c26:c31) / SUM(c26:c31)
```
This computes the weighted mean of the Net reliability gap percentages using Compute Capacity as weights.

## Step 4 – Save and Validate
1. Save the workbook to `/root/output/result.xlsx` (use `openpyxl` save; do not alter formatting, do not add sheets/macros/VBA).
2. Re-open the saved file and print sample formulas from each block to confirm:
   - Lookup formulas use INDEX/MATCH with correct mixed references.
   - H35:L40 formulas have row-relative references (H35 ≠ H36 etc.).
   - H42:L47 contain the six summary statistic formulas.
   - H50:L50 contain SUMPRODUCT-based weighted means.
3. Confirm no extra sheets were added and the file is well-formed.

## Important Notes
- Before writing any formula, always read the current cell/range to confirm it is empty or yellow-targeted.
- After writing formulas, re-read them to confirm they landed correctly.
- Do NOT use `data_only=True` when loading – that would strip formulas.
- Do NOT add helper columns, helper sheets, macros, or VBA.
- Preserve all existing formatting.

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