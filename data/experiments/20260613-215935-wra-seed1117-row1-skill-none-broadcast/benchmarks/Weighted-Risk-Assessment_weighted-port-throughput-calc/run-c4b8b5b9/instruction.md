# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 – Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1 – Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and print:
- Sheet names.
- `Task` sheet: cells D12:D17, D19:D24, D26:D31 (series codes), row 10 columns H–L (years), cells H35:L40 labels/current content, H42:H47 labels, H50 label, and any existing content in the yellow target ranges.
- `Data` sheet: row 21 headers and a sample of rows 21–38, noting the column layout (which column holds the series code, which columns hold years/values).

This inspection is critical — do NOT skip it. Print enough to understand the exact column letters and row numbers on `Data` so that MATCH references are correct.

## 2 – Write a Python script that populates formulas

Using openpyxl, load the workbook (keep formatting), and write **string formulas** (not computed values) into every target cell. Do NOT use `data_only=True`.

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell at row `r`, column `c` (H=8 … L=12):
- The series code is in `Task!D{r}`.
- The year is in `Task!{col_letter}10` (where col_letter matches c).
- The data lives in `Data!$A$21:$<last_col>$38` (adjust column letters based on your inspection).
- Use the pattern: `=INDEX(Data!<value_range>, MATCH(D{r}, Data!<series_code_column>, 0), MATCH({col_letter}10, Data!<year_header_row>, 0))`
  — Adjust the ranges precisely based on what you found in step 1. The MATCH for the series code should search the single column of series codes in Data rows 21–38. The MATCH for the year should search the header row of Data that contains year values.

### Step 2a – Net container flow in H35:L40
For each cell at row `r_net` (35–40) and column `c`:
- Identify which rows in the earlier blocks correspond to Loaded Containers Inbound, Loaded Containers Outbound, and Terminal Throughput Capacity. Based on the three blocks (H12:L17 = block1, H19:L24 = block2, H26:L31 = block3), determine which block is which metric by inspecting the labels in column B or C of the Task sheet.
- Formula: `=(<Inbound_cell> - <Outbound_cell>) / <Capacity_cell> * 100`
  where each cell reference is the same column `c` and the corresponding row within its block that matches the same port (same relative position: 1st port = offset 0, 2nd = offset 1, etc.).

### Step 2b – Summary statistics in H42:L47
For each column `c`:
- H42:L42 → `=MIN({col}35:{col}40)`
- H43:L43 → `=MAX({col}35:{col}40)`
- H44:L44 → `=MEDIAN({col}35:{col}40)`
- H45:L45 → `=AVERAGE({col}35:{col}40)`
- H46:L46 → `=PERCENTILE({col}35:{col}40, 0.25)`
- H47:L47 → `=PERCENTILE({col}35:{col}40, 0.75)`

**Important**: Check the labels in column B/C/D of rows 42–47 to confirm the order (min, max, median, mean, 25th, 75th). If the order differs, match the formula to the label, not the row number.

### Step 3 – Weighted mean in H50:L50
For each column `c`:
- `=SUMPRODUCT({col}35:{col}40, {col}26:{col}31) / SUM({col}26:{col}31)`
  (This is the weighted mean using Terminal Throughput Capacity as weights.)

**Verify that H26:L31 is indeed the Terminal Throughput Capacity block.** If inspection shows it's a different block, adjust accordingly.

## 3 – Save and verify
Save to `/root/output/result.xlsx`.

Then reload the file and print every target cell's `.value` (the formula string) to confirm:
- No cell in H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50 is None or empty.
- All cells contain formula strings starting with `=`.
- The Data sheet references use correct column letters and row numbers.

## Critical Checks
- **Do NOT leave any target cell empty or None.** This was the failure mode in the hospital-bedflow task.
- **Do NOT add new sheets, macros, or VBA.**
- **Do NOT alter existing formatting** — only write formula strings into the target cells.
- **Print the formulas you wrote** so you can visually verify correctness before finishing.

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