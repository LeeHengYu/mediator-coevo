# Task Instruction

## Task: Weighted-Risk-Assessment/hospital-capacity-at-risk-calc

You must update `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Do NOT add sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

### Preparation

1. `mkdir -p /root/output`
2. Install openpyxl if not already available: `pip install openpyxl`
3. Open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: read the layout of rows 10-50, columns D and H-L. Identify:
     - Row 10: the year headers in H10:L10
     - Column D rows 12-17, 19-24, 26-31: the series codes
     - H12:L17, H19:L24, H26:L31: the yellow cells needing lookup formulas
     - H35:L40: Net capacity headroom calculation cells
     - H42:L47: summary statistics cells (min, max, median, mean, 25th pctl, 75th pctl)
     - H50:L50: weighted mean cells
   - Sheet `Data`: inspect rows 21-38 to understand the data layout (which row holds headers, which column holds series codes, which columns hold year data). Determine whether data is arranged with series codes in a column and years across columns (suitable for VLOOKUP/INDEX-MATCH) or vice versa.
4. Print out the exact cell contents of Task!D12:D31, Task!H10:L10, and Data rows 21-38 (all used columns) so you understand the structure before writing any formulas.

### Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write an Excel formula using `INDEX` with `MATCH` (preferred for reliability). The formula should:
- Use the series code from column D of the **same row** (e.g., `$D12` for row 12, with $ on column to allow horizontal copying)
- Use the year from row 10 of the **same column** (e.g., `H$10` for column H, with $ on row)
- Look up in `Data!` rows 21:38. Determine the exact range based on your inspection:
  - If series codes are in a column (say column A or B of Data), use that as the lookup column
  - The data values span across columns corresponding to years
  - Pattern: `=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))`
- Adjust ranges precisely based on what you observe in the Data sheet.

IMPORTANT: Inspect the Data sheet carefully. The rows 21:38 instruction means the relevant data is in that row range. Find where series codes live and where year headers live within or adjacent to that range.

### Step 2: Net Capacity Headroom (H35:L40) and Summary Statistics (H42:L47)

For H35:L40 (6 rows × 5 columns):
- Formula: `(Available Care Slots - Occupied Care Slots) / Staffed Bed Capacity * 100`
- Identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to Available Care Slots, Occupied Care Slots, and Staffed Bed Capacity by reading labels in the Task sheet (likely in column C, D, E, F, or G near those blocks, or in a header row above each block).
- For each row i (1-6 representing the six hospital clusters) and column j (H-L representing years), the formula references the corresponding cells from those three blocks.

For H42:L47 (column-wise statistics over H35:L40):
- Row 42: MIN of H35:H40 (for column H), etc.
- Row 43: MAX of H35:H40, etc.
- Row 44: MEDIAN of H35:H40, etc.
- Row 45: AVERAGE of H35:H40, etc.
- Row 46: PERCENTILE(H35:H40, 0.25), etc.
- Row 47: PERCENTILE(H35:H40, 0.75), etc.
- IMPORTANT: Check the labels in column D/E/F/G for rows 42-47 to confirm which row gets which statistic. Map them exactly.

### Step 3: Weighted Mean in H50:L50

For each column (H through L):
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
This computes the weighted mean of the Net capacity headroom percentages weighted by Staffed Bed Capacity.

### Implementation Approach

Use openpyxl to write formulas as strings into cells. Do NOT compute values in Python — write Excel formula strings so the spreadsheet recalculates.

Example:
```python
ws['H12'] = '=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))'
```
(Adjust ranges based on actual inspection.)

After writing all formulas, save to `/root/output/result.xlsx`.

### Validation

1. Re-open the saved file with openpyxl and verify that cells H12, L31, H35, H47, H50 contain formula strings (start with '=').
2. Verify no new sheets were added.
3. Verify the file is valid xlsx by loading it without errors.
4. Print a sample of formulas from each section to confirm correctness.

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