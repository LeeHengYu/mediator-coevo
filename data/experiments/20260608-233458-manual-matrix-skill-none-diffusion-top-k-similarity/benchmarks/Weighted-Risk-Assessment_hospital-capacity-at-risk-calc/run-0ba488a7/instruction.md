# Task Instruction

## Task: Update hospital capacity workbook with formulas

### Overview
You must read, understand, and update `/root/data/workbook.xlsx` by populating specific cells with spreadsheet formulas, then save to `/root/output/result.xlsx`.

### Step 0: Inspect the workbook
1. `mkdir -p /root/output`
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Read the structure carefully. Print rows 1-55 or so, paying special attention to:
     - Column D (series codes for each row)
     - Row 10 (years in columns H through L)
     - The yellow cell ranges: H12:L17, H19:L24, H26:L31
     - H35:L40, H42:L47, H50:L50
   - Sheet `Data`: Read rows 21-38 carefully. Print the full content including headers. Understand the layout — is the data organized with series codes as row keys and years as column headers, or vice versa? Identify exactly which row/column contains series codes and which contains years.
3. Print cell fill colors for a few yellow cells to confirm which cells need formulas.
4. Understand the exact text of series codes in column D of `Task` sheet and how they correspond to entries in `Data` sheet rows 21:38.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these ranges, write a spreadsheet formula (not a Python-computed value) that:
- Takes two inputs: the series code from column D of that row, and the year from row 10 of that column
- Looks up the value from sheet `Data` rows 21:38
- Uses one of the allowed patterns: `INDEX(MATCH,MATCH)`, `VLOOKUP+MATCH`, `HLOOKUP+MATCH`, or `XLOOKUP+MATCH`

IMPORTANT FORMULA DETAILS:
- You must write actual Excel formulas as strings starting with `=`
- Reference the `Data` sheet appropriately (e.g., `Data!A21:A38` or similar)
- Use `$` signs for anchoring where appropriate (anchor the lookup range and header ranges, but let the row's series code reference and column's year reference vary)
- The series code reference should be anchored to column D but vary by row (e.g., `$D12`)
- The year reference should be anchored to row 10 but vary by column (e.g., `H$10`)
- Inspect the Data sheet layout carefully to determine if you need a row-based or column-based lookup. For INDEX/MATCH, you'd typically use two MATCH functions — one for the row dimension and one for the column dimension.
- Make sure match_type is 0 (exact match)

### Step 2: Net capacity headroom in H35:L40
For each cell in H35:L40, write a formula:
`= (AvailableCareSlots - OccupiedCareSlots) / StaffedBedCapacity * 100`

where:
- `AvailableCareSlots` comes from the corresponding cell in H12:L17
- `OccupiedCareSlots` comes from the corresponding cell in H19:L24  
- `StaffedBedCapacity` comes from the corresponding cell in H26:L31

So for H35: `=(H12-H19)/H26*100`, for I35: `=(I12-I19)/I26*100`, etc.
For H36: `=(H13-H20)/H27*100`, etc.

Then in H42:L47, write formulas for column-wise statistics over H35:L40:
- Row 42: MIN (e.g., `=MIN(H35:H40)`)
- Row 43: MAX (e.g., `=MAX(H35:H40)`)
- Row 44: MEDIAN (e.g., `=MEDIAN(H35:H40)`)
- Row 45: AVERAGE (e.g., `=AVERAGE(H35:H40)`)
- Row 46: 25th percentile (e.g., `=PERCENTILE(H35:H40,0.25)`)
- Row 47: 75th percentile (e.g., `=PERCENTILE(H35:H40,0.75)`)

IMPORTANT: Verify which rows correspond to MIN, MAX, MEDIAN, MEAN, 25th, 75th by checking any labels in the Task sheet. The order above is my best guess — adjust based on actual labels in column D or nearby columns for rows 42-47.

### Step 3: Weighted mean in H50:L50
For each column H through L, write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net capacity headroom percentages (H35:H40) weighted by Staffed Bed Capacity (H26:H31).

### Step 4: Save
- Save the workbook to `/root/output/result.xlsx`
- Do NOT change formatting, do NOT add sheets, macros, VBA, external links, or helper tabs
- Use `openpyxl` and be careful to preserve existing formatting by loading with `keep_vba=False` and without `data_only=True`

### Validation
After saving:
1. Reopen `/root/output/result.xlsx` and verify that cells H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50 all contain formula strings (starting with `=`)
2. Print a sample of formulas to confirm they reference the correct cells and sheets
3. Confirm no new sheets were added
4. Confirm the formulas use one of the allowed lookup patterns for Step 1

### Critical Notes
- You MUST inspect the Data sheet thoroughly before writing formulas. The exact cell references in your formulas depend on the actual layout.
- All formulas must be Excel formulas (strings), not computed Python values.
- Preserve all existing content and formatting. Only write to the specified yellow cells.
- Use `openpyxl` — do not use xlsxwriter (it cannot modify existing files).

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