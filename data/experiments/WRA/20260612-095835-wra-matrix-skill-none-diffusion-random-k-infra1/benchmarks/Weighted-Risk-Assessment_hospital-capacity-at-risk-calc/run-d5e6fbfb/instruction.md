# Task Instruction

Execute the following steps precisely to complete the hospital capacity risk assessment workbook.

## 0. Setup and Inspection

1. `mkdir -p /root/output`
2. `cp /root/data/workbook.xlsx /root/output/result.xlsx`
3. Open `/root/output/result.xlsx` with openpyxl (with `data_only=False` so you preserve and write formulas).
4. Inspect sheet `Task`: print the contents of columns A–L for rows 1–55. Pay special attention to:
   - Row 10 (the year headers in H10:L10)
   - Column D rows 12–31 (the series codes)
   - The structure of H12:L17, H19:L24, H26:L31 (the yellow lookup cells)
   - Row 35–40 labels and any existing content
   - Row 42–47 labels (min, max, median, mean, 25th, 75th percentile)
   - Row 50 label
5. Inspect sheet `Data`: print rows 1–40 focusing on rows 21–38. Identify:
   - Which row contains the series codes (likely column A or B)
   - Which row contains the year headers
   - The exact layout so you know the lookup table dimensions
6. Print all findings before writing any formulas.

## 1. Step 1 – Lookup Formulas in H12:L31

For each cell in the three blocks (H12:L17, H19:L24, H26:L31), write an INDEX/MATCH formula that:
- Uses the series code from column D of that row
- Uses the year from row 10 of that column
- Looks up in sheet `Data` rows 21:38

The exact formula pattern depends on the Data sheet layout. Assuming the series codes are in one column (say column A) and years are in a header row (say row 20), the formula pattern would be:

`=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))`

Adjust the ranges based on what you discover in step 0. The key constraints:
- The row lookup must match on column D of the current Task sheet row (use $ to lock the column: `$D12`)
- The column lookup must match on row 10 of the current Task sheet column (use $ to lock the row: `H$10`)
- The data range on sheet Data must cover rows 21:38 and the corresponding columns
- Use absolute references for the Data ranges so the formula copies correctly across the 5 columns and 6 rows of each block

Write the formula into every cell in all three blocks (H12:L17, H19:L24, H26:L31) — that's 90 cells total (3 blocks × 6 rows × 5 columns).

## 2. Step 2 – Net Capacity Headroom (H35:L40)

The formula for each cell in H35:L40 is:
`=(H12 - H19) / H26 * 100`

More precisely, for cell H35:
`=(H12-H19)/H26*100`

Where:
- Row 12–17 = Available Care Slots (first block)
- Row 19–24 = Occupied Care Slots (second block)  
- Row 26–31 = Staffed Bed Capacity (third block)

So H35 references H12, H19, H26; H36 references H13, H20, H27; etc. Adjust row offsets accordingly. The pattern for cell in row r, column c of the headroom block (r=35..40, corresponding to data rows offset 0..5):
`=(Cx(r-23) - Cx(r-16)) / Cx(r-9) * 100`

Verify the row arithmetic: row 35 → data rows 12, 19, 26; row 36 → 13, 20, 27; etc.

## 3. Step 2 continued – Summary Statistics (H42:L47)

For each column (H through L), compute these over the 6 headroom values (e.g., H35:H40):
- H42: `=MIN(H35:H40)`
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)`
- H46: `=PERCENTILE(H35:H40,0.25)`
- H47: `=PERCENTILE(H35:H40,0.75)`

Check the row labels in column A/B/C to confirm which row is which statistic. Adjust the row assignments if the labels indicate a different order.

## 4. Step 3 – Weighted Mean (H50:L50)

For each column (H through L):
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the headroom percentages using Staffed Bed Capacity as weights.

## 5. Save and Validate

1. Save the workbook with `wb.save('/root/output/result.xlsx')`.
2. Reopen the file and verify:
   - Cells H12:L31 contain INDEX/MATCH formulas (not hardcoded values)
   - Cells H35:L40 contain arithmetic formulas
   - Cells H42:L47 contain statistical function formulas
   - Cells H50:L50 contain SUMPRODUCT formulas
   - No new sheets were added
   - No macros or VBA
3. Print a sample of formulas from each block to confirm correctness.

## Critical Constraints
- Do NOT use `data_only=True` when loading — you must preserve existing formulas.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change any existing formatting.
- Only write to the specified cell ranges.
- Use openpyxl to write formula strings (e.g., `ws['H12'] = '=INDEX(...)'`).
- All references to the Data sheet in formulas must use the syntax `Data!` prefix.
- Lock references appropriately with `$` signs so formulas work when conceptually "copied" across the block (even though you're writing each cell individually, the pattern should be consistent).

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