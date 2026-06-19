# Task Instruction

Execute the following steps carefully to produce `/root/output/result.xlsx`.

## Phase 0 – Inspect the workbook structure

1. Copy the source workbook:
   ```
   mkdir -p /root/output
   cp /root/data/workbook.xlsx /root/output/result.xlsx
   ```

2. Open `/root/output/result.xlsx` with openpyxl (data_only=False) and print:
   - Sheet names.
   - On sheet `Task`: the contents of cells D12:D17, D19:D24, D26:D31 (series codes), row 10 columns H–L (years), cells G12:G17, G19:G24, G26:G31 (row labels), and any existing content in H12, H35, H42, H50.
   - On sheet `Data`: print rows 19–40 (all columns with data) so you can see the full layout — which column holds series codes, which row holds years, and where numeric data starts. Also print the header row (row 1 or whichever row contains column headers).
   - Print the exact column letters and row numbers so formulas can reference them precisely.

## Phase 1 – Write lookup formulas (H12:L31)

Based on the inspection, write INDEX/MATCH formulas into cells H12:L31 on sheet `Task`. The pattern for each cell should be:

```
=INDEX(Data!<data_range>, MATCH(<series_code_cell>, Data!<series_code_column>, 0), MATCH(<year_cell>, Data!<year_row>, 0))
```

Critical details:
- `<series_code_cell>` is the absolute reference to column D of the current row on `Task` (e.g., `$D12`).
- `<year_cell>` is the absolute reference to the year in row 10 for the current column (e.g., `H$10`).
- `<data_range>` must be the rectangular block on `Data` that contains the numeric values (rows 21–38, and the columns that hold the yearly data). Determine the exact columns from inspection.
- `<series_code_column>` is the single column on `Data` that holds the series codes, spanning rows 21–38.
- `<year_row>` is the single row on `Data` that holds the year headers, spanning the same columns as the data range.
- Use absolute references (`$`) appropriately so the formula can be written per-cell correctly. The series code reference should lock the column (`$D12`), and the year reference should lock the row (`H$10`).

Write formulas for all three blocks: H12:L17, H19:L24, H26:L31.

## Phase 2 – Net patient flow (H35:L40)

For each hospital (rows 35–40) and each year column (H–L), write a formula:
```
=(H12 - H19) / H26 * 100
```
where H12 is the Patient Admissions cell, H19 is the Patient Discharges cell, and H26 is the Effective Bed Capacity cell for the same hospital and year. Adjust row references for each hospital row (35→12,19,26; 36→13,20,27; etc.).

## Phase 3 – Statistics (H42:L47)

For each year column (H–L), write these formulas in the six statistic rows:
- Row 42 (Min): `=MIN(H35:H40)`
- Row 43 (Max): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

IMPORTANT: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to ensure compatibility. The function name in Excel is `PERCENTILE` for the classic version. If you saw #NAME? errors in a prior run, this is likely the cause.

## Phase 4 – Weighted mean (H50:L50)

For each year column (H–L), write:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of net patient flow using Effective Bed Capacity as weights.

## Phase 5 – Save and validate

1. Save the workbook with openpyxl.
2. Re-open the saved file (data_only=False) and print the formula strings in cells H12, L17, H35, L40, H42, H47, H50, L50 to confirm they look correct.
3. Then re-open with data_only=True and check if any cells return None (which is expected for formula cells not yet evaluated by Excel — this is fine for openpyxl).
4. Optionally, use a quick Python evaluation: load with data_only=False, parse a couple of the INDEX/MATCH formulas manually to verify the references point to real data on the Data sheet.

## Key Warnings
- Do NOT use `PERCENTILE.INC` or `PERCENTILE.EXC` — use `PERCENTILE`.
- Do NOT add any new sheets, macros, VBA, or external links.
- Do NOT change existing formatting.
- Make sure the Data sheet references are correct by inspecting the actual layout first. The series codes and year headers must match exactly.
- When writing formulas with openpyxl, prefix with `=` and use comma `,` as the argument separator (openpyxl uses Excel-style A1 notation with commas).
- Double-check that the Data sheet range in INDEX covers both the correct rows (21–38) and the correct columns (the ones holding numeric data, NOT the series code column).

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