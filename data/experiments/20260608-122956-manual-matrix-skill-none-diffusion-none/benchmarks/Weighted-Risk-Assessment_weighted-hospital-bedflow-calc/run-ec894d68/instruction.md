# Task Instruction

## Task: Weighted Hospital Bedflow Calculation

You must update `/root/data/workbook.xlsx` by populating formulas in the `Task` sheet, then save to `/root/output/result.xlsx`.

### Preliminary Investigation

1. First, read the workbook to understand its structure:
   - Open `/root/data/workbook.xlsx` using openpyxl.
   - Inspect sheet names (should have `Task` and `Data`).
   - On sheet `Task`: print rows 10-50 to understand the layout, especially:
     - Column D (series codes) for rows 12-17, 19-24, 26-31
     - Row 10 (years in columns H through L)
     - Row 35-40 labels and any existing content
     - Rows 42-47 labels (min, max, median, mean, 25th, 75th percentile)
     - Row 50 label
   - On sheet `Data`: print rows 21-38 to understand the lookup source structure. Note which row contains headers, which column has series codes, and how the data is arranged (row-wise vs column-wise).
   - Print cell fills/colors if possible to confirm yellow cells.

2. Determine the exact layout of the `Data` sheet rows 21:38:
   - Identify where the series codes are and where the year-based values are.
   - Determine whether data is arranged for VLOOKUP (series code in first column, years across columns) or needs a different approach.

### Step 1: Populate Lookup Formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, create a formula that:
- Uses the series code from column D of that row
- Uses the year from row 10 of that column (H10, I10, J10, K10, L10)
- Looks up the value from `Data` sheet rows 21:38

Use INDEX/MATCH pattern (most flexible). The exact formula depends on the Data sheet layout you discover. A likely pattern is:
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
But you MUST adjust the ranges based on actual inspection of the Data sheet. The key requirements:
- The row match should find the series code from column D in the appropriate column of Data rows 21:38
- The column match should find the year from row 10 in the appropriate header row of the Data sheet
- Use absolute references appropriately so formulas can span the range
- Use one of these patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH

IMPORTANT: When writing formulas with openpyxl, do NOT include the leading `=` sign if using `cell.value = '=FORMULA...'` — actually, openpyxl DOES require the leading `=`. Set cell values like: `cell.value = '=INDEX(...)'`

### Step 2: Net Patient Flow (H35:L40) and Summary Statistics (H42:L47)

For H35:L40, the formula for each hospital/year cell is:
```
=(PatientAdmissions - PatientDischarges) / EffectiveBedCapacity * 100
```
where:
- Patient Admissions are in H12:L17 (rows 12-17)
- Patient Discharges are in H19:L24 (rows 19-24)  
- Effective Bed Capacity are in H26:L31 (rows 26-31)

So for cell H35: `=(H12-H19)/H26*100`
For cell H36: `=(H13-H20)/H27*100`
And so on for each hospital row and year column.

Wait — verify the exact row mapping by checking that rows 12-17 correspond to the same hospitals as rows 19-24, 26-31, and 35-40. The offset between blocks should be consistent.

For H42:L47 (column-wise statistics over H35:L40):
- H42: `=MIN(H35:H40)` (minimum)
- H43: `=MAX(H35:H40)` (maximum)
- H44: `=MEDIAN(H35:H40)` (median)
- H45: `=AVERAGE(H35:H40)` (simple mean)
- H46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- H47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

Check the labels in column D/E/F/G for rows 42-47 to confirm which row is which statistic and adjust accordingly.

### Step 3: Weighted Mean (H50:L50)

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of net patient flow percentages weighted by effective bed capacity.

### Final Steps

1. Ensure you do NOT modify any existing formatting. Use openpyxl with `keep_vba=False` and load with `data_only=False` to preserve formulas.
2. Do NOT add any new sheets, macros, or external links.
3. Create `/root/output/` directory if it doesn't exist.
4. Save to `/root/output/result.xlsx`.

### Validation

After saving, reload the file and verify:
- Cells H12:L31 contain formula strings (not None or static values)
- Cells H35:L40 contain formula strings
- Cells H42:L47 contain formula strings
- Cells H50:L50 contain formula strings
- Sheet names are unchanged (only `Task` and `Data`)
- Print a sample of formulas to confirm correctness

### Critical Notes
- You MUST inspect the actual workbook structure before writing any formulas. The exact cell references in formulas depend entirely on how the Data sheet is laid out.
- Use `openpyxl` for reading and writing.
- When loading, do NOT use `data_only=True` (that would strip formulas).
- Preserve existing cell styles by not overwriting them — just set `.value` on cells that need formulas.
- The row-to-row mapping between the three blocks (admissions rows 12-17, discharges rows 19-24, capacity rows 26-31) and the net flow block (rows 35-40) must correspond to the same hospitals. Verify this by checking column D labels.

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