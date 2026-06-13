# Task Instruction

Execute the following steps exactly to produce /root/output/result.xlsx.

## 0 — Setup
```bash
mkdir -p /root/output
```

## 1 — Inspect the workbook
Open /root/data/workbook.xlsx with openpyxl (data_only=False). On the **Task** sheet:
- Print rows 10-11 (columns A-L) to see the year headers in row 10.
- Print rows 12-31 (columns A-L) to see the series codes in column D and the yellow target ranges.
- Print rows 35-50 (columns A-L) to see the labels for Net renewable balance, statistics rows, and MCEC row.

On the **Data** sheet:
- Print rows 19-40 (columns A-Z or however far the data extends) to understand the layout: which row holds headers, which column holds series codes, and where years appear.
- Identify the exact row range (21:38) and the column layout so you know how MATCH will resolve.

Record:
- The year values in Task!H10:L10.
- The series codes in Task!D12:D17, D19:D24, D26:D31.
- The header row and series-code column on the Data sheet.
- The data extent on the Data sheet.

## 2 — Write lookup formulas into H12:L17, H19:L24, H26:L31
Using openpyxl, write **Excel formulas** (not computed values) into each cell.

Use the INDEX/MATCH/MATCH pattern with mixed references so a single formula template can be applied across the 5×6 block:

```
=INDEX(Data!$A$21:$ZZ$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$A$21:$ZZ$21, 0))
```

Adjust the data range columns based on what you discovered in Step 1 (e.g., if Data columns go A through say T, use Data!$A$21:$T$38 etc.). The key references:
- Row anchor: `$D12` (column D is absolute, row is relative — changes per row)
- Column anchor: `H$10` (row 10 is absolute, column is relative — changes per column)
- The lookup array for rows is the series-code column in Data (e.g., Data!$A$21:$A$38).
- The lookup array for columns is the header row in Data (e.g., Data!$A$21:$T$21 or whichever row contains the year headers — verify whether it's row 21 or another row).

IMPORTANT: Verify which row on the Data sheet contains the year headers that MATCH should search against. It might be row 20 or row 21. Adjust accordingly. The INDEX range and the column-MATCH range must be consistent.

Write these formulas for all three blocks:
- H12:L17 (6 rows × 5 cols = 30 cells)
- H19:L24 (30 cells)
- H26:L31 (30 cells)

Total: 90 lookup cells.

## 3 — Write Net renewable balance formulas in H35:L40
For each campus (6 rows) and each year (5 columns), the formula is:
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to Renewable Generation, H19 to Grid Consumption, H26 to Baseline Energy Demand — adjusted by row offset for each campus. Verify which block maps to which concept by reading the labels in column A/B of the Task sheet. The mapping should be:
- Block H12:L17 → first data concept (check label)
- Block H19:L24 → second data concept (check label)
- Block H26:L31 → third data concept (check label)

The formula for H35 should reference the first row of each block for the same campus, e.g.:
```
=(H12 - H19) / H26 * 100
```
H36: `=(H13 - H20) / H27 * 100` etc.

Use relative references so they shift correctly across columns.

## 4 — Write statistics formulas in H42:L47
Based on the row labels (inspect them), write column-wise formulas over H35:H40 (the 6 campus values). Expected mapping (verify labels):
- Row 42: `=MIN(H35:H40)` (or whichever label says minimum)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

Match each formula to the actual label in column A/B of that row. Do NOT assume the order above — read the labels first.

## 5 — Write MCEC weighted mean in H50:L50
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the net renewable balance percentages using Baseline Energy Demand as weights. Write this for columns H through L.

## 6 — Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change any formatting, do not add sheets, macros, VBA, or external links.

## 7 — Validate
Reopen `/root/output/result.xlsx` with openpyxl (data_only=False) and:
- Print cells H12, L17, H19, L24, H26, L31 to confirm they contain formula strings (not None).
- Print cells H35, L40 to confirm they contain formula strings.
- Print cells H42:L47 to confirm they contain formula strings.
- Print cells H50:L50 to confirm they contain formula strings.
- Confirm no cell in the target ranges is None.

If any cell is None, investigate and fix before finishing.

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