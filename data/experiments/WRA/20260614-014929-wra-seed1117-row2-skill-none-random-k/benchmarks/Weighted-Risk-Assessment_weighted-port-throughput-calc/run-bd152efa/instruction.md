# Task Instruction

## Task: Update /root/data/workbook.xlsx with formulas and save to /root/output/result.xlsx

### Phase 0: Inspection

1. Create the output directory: `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` using openpyxl and inspect:
   - Sheet `Task`: Read the layout carefully.
     - Column D rows 12-17, 19-24, 26-31: note the series codes.
     - Row 10 columns H-L: note the year headers.
     - Rows 35-40: note which ports correspond to which rows.
     - Rows 42-47: note the labels (min, max, median, mean, 25th pctile, 75th pctile).
     - Row 50: note the CPA weighted mean label.
   - Sheet `Data`: Read rows 21-38 to understand the data layout (which row is the header, which column has series codes, where years appear).
   - Print all of this information so you can construct correct formulas.

### Phase 1: Populate formulas using openpyxl

Use `openpyxl` to write formulas into the cells. Do NOT use `data_only` mode. Preserve all existing formatting by loading with `load_workbook(filename, data_only=False)`. Do NOT create new sheets.

#### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write an `INDEX/MATCH` formula that:
- Uses the series code from column D of that row (use mixed reference like `$D12` so column is fixed)
- Uses the year from row 10 of that column (use mixed reference like `H$10` so row is fixed)
- Looks up data from the `Data` sheet rows 21-38
- Pattern: `=INDEX(Data!$B$22:$XX$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$XX$21, 0))`
- Adjust the exact column/row references based on what you discover in Phase 0 about the Data sheet layout (which column has series codes, which row has year headers, and the extent of the data range).

#### Step 2: Net container flow in H35:L40

The formula for each cell should be:
`=(H12 - H19) / H26 * 100`
where the row references correspond to the matching port rows:
- Row 35 uses rows 12, 19, 26
- Row 36 uses rows 13, 20, 27
- Row 37 uses rows 14, 21, 28
- Row 38 uses rows 15, 22, 29
- Row 39 uses rows 16, 23, 30
- Row 40 uses rows 17, 24, 31

Use relative references so each column (H-L) picks up its own year.

#### Step 2 continued: Statistics in H42:L47

For each column H through L:
- H42: `=MIN(H35:H40)`
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)`
- H46: `=PERCENTILE(H35:H40, 0.25)`
- H47: `=PERCENTILE(H35:H40, 0.75)`

**CRITICAL**: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid #NAME? errors. Verify the row labels in Phase 0 to confirm the correct ordering (min, max, median, mean, 25th, 75th) — adjust if the labels differ from what's assumed here.

#### Step 3: Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the net container flow percentages using Terminal Throughput Capacity as weights.

### Phase 2: Save and Verify

1. Save the workbook to `/root/output/result.xlsx`.
2. Reopen the saved file and verify:
   - All formula cells in the target ranges contain formula strings (start with '=').
   - No cells are empty or contain plain values where formulas are expected.
   - The formulas reference the correct sheets and ranges.
   - Print a sample of formulas from each section for confirmation.

### Important Constraints
- Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify any existing formatting.
- Do NOT use `data_only=True` when loading.
- Verify the actual row/column layout in Phase 0 before writing any formulas — the exact references above are estimates that must be confirmed against the actual workbook structure.

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