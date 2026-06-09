# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 — Preparation
```bash
mkdir -p /root/output
```
Open and inspect `/root/data/workbook.xlsx` with openpyxl (data_only=False) to confirm:
- Sheet names: `Task` and `Data`.
- On `Task`: column D contains series codes starting from row 12; row 10 contains year headers in columns H–L; yellow target ranges are H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50.
- On `Data`: rows 21–38 hold the source data; identify which column holds the series code (the lookup key) and which row holds the year headers so you know the orientation of the data table.

Print out:
- `Task` sheet: cells D12:D17, D19:D24, D26:D31 (series codes), H10:L10 (years), and row labels for rows 35–50.
- `Data` sheet: the header row for the data block and a few sample rows (rows 21–38) to confirm layout (especially the first column = series code, and the header row that contains years).

## 1 — Step 1: Lookup formulas in H12:L31

For each cell in the three blocks (H12:L17, H19:L24, H26:L31), write an INDEX/MATCH formula. The exact formula pattern depends on the Data sheet layout you discovered above, but it should follow this template:

```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Concrete example (adjust ranges after inspection):
- If Data sheet has series codes in column A rows 21:38 and year headers in row 20 columns B onward, and data fills B21:?38, then for cell H12:
  ```
  =INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
  ```
- Use `$D12` (column absolute, row relative) and `H$10` (column relative, row absolute) so the formula can be filled across the entire block.

Write these formulas programmatically with openpyxl by iterating over the three row-ranges and columns H(8) through L(12).

## 2 — Step 2a: Net capacity headroom (H35:L40)

The six rows correspond to six hospital clusters. Based on the three lookup blocks:
- Block 1 (rows 12–17): one metric per cluster
- Block 2 (rows 19–24): another metric per cluster
- Block 3 (rows 26–31): another metric per cluster

Identify which block is "Available Care Slots", "Occupied Care Slots", and "Staffed Bed Capacity" by reading the labels in the Task sheet (likely in column B or C for each block header). Then for each cell in H35:L40:
```
=(H12 - H19) / H26 * 100
```
(Adjust row references based on which block maps to which metric. The pattern: first cluster in row 35 uses data from rows 12, 19, 26; second cluster in row 36 uses rows 13, 20, 27; etc.)

Use relative references so the formula naturally fills across the 6×5 block.

## 2b — Summary statistics (H42:L47)

For each column H through L, compute six statistics over the six headroom values in rows 35–40. Map the six rows to these formulas:
- Row 42 (MIN):    `=MIN(H35:H40)`
- Row 43 (MAX):    `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN):   `=AVERAGE(H35:H40)`
- Row 46 (25th %): `=PERCENTILE.INC(H35:H40, 0.25)`
- Row 47 (75th %): `=PERCENTILE.INC(H35:H40, 0.75)`

**IMPORTANT (from cross-task feedback):** Verify the exact row-to-statistic mapping by reading the labels in the Task sheet for rows 42–47 before assigning formulas. The order may differ. Also, use `PERCENTILE.INC` (with the dot) — this is the correct modern Excel function name. Do NOT use `PERCENTILE` without the suffix, and do NOT use `_xlfn.` prefix (openpyxl handles the prefix internally when you use the dotted form in newer versions, but test this). If openpyxl writes `_xlfn.PERCENTILE.INC` automatically that is fine; if it doesn't and you get #NAME? errors, you may need to explicitly prefix with `_xlfn.` — but first try without.

Actually, to be safe and avoid the #NAME? issue from the cross-task context: when writing formulas with openpyxl for functions like `PERCENTILE.INC` and `MEDIAN`, explicitly use the `_xlfn.` prefix:
- `=_xlfn.PERCENTILE.INC(H35:H40, 0.25)` for 25th percentile
- `=_xlfn.PERCENTILE.INC(H35:H40, 0.75)` for 75th percentile  
- `MEDIAN` → use `=MEDIAN(H35:H40)` (MEDIAN usually works without prefix, but if issues arise, try `=_xlfn.MEDIAN(H35:H40)`)
- `MIN`, `MAX`, `AVERAGE` don't need prefixes.

## 3 — Step 3: Weighted mean (H50:L50)

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net capacity headroom percentages using Staffed Bed Capacity as weights.

## 4 — Save

Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets, macros, or external links.

## 5 — Validation

After saving, reopen `/root/output/result.xlsx` with openpyxl (data_only=False) and print:
- A sample lookup cell (e.g., H12) to confirm it contains a formula string.
- A sample headroom cell (e.g., H35) to confirm formula.
- All six stat cells in column H (H42:H47) to confirm formula strings and especially that PERCENTILE.INC formulas are present.
- H50 formula.
- Confirm sheet names are exactly `Task` and `Data` with no extra sheets.

If any formula shows a raw `_xlfn.` prefix that shouldn't be there, or if a function name looks wrong, fix it before finalizing.

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