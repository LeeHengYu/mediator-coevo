# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

### Overview
You need to open `/root/data/workbook.xlsx`, add spreadsheet formulas to specific cells on the `Task` sheet, and save the result to `/root/output/result.xlsx`. Do NOT add sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

### Before You Start
1. Inspect the workbook structure thoroughly:
   - Read the `Task` sheet to understand the layout: column D series codes, row 10 years, the yellow cell ranges, and any existing content.
   - Read the `Data` sheet rows 21:38 to understand the source data structure (column headers, row labels, how series codes and years are arranged).
   - Identify exactly how series codes and years map between `Task` and `Data` sheets.
2. Check what libraries are available (openpyxl is likely the tool). Note: openpyxl writes formula strings that Excel/LibreOffice evaluates. You are writing *formula text*, not computing values in Python.

### Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write a formula that looks up data from `Data!$21:$38` using:
- The series code from column D of the current row on `Task` sheet
- The year from row 10 of the corresponding column on `Task` sheet

Use `INDEX` with `MATCH` as the lookup pattern (it's the most robust). The formula pattern should be something like:
```
=INDEX(Data!<data_range>, MATCH(<series_code_ref>, Data!<series_code_column>, 0), MATCH(<year_ref>, Data!<year_row>, 0))
```

IMPORTANT: Inspect the Data sheet carefully to determine:
- Which column contains the series codes (could be column A, B, C, etc.)
- Which row contains the year headers
- What the actual data range is
- Whether series codes are text strings and need exact match

Anchor references appropriately (e.g., the series code column reference should lock the column, the year row reference should lock the row).

### Step 2: Net Patient Flow in H35:L40 and Statistics in H42:L47

**H35:L40 - Net Patient Flow:**
For each of the six hospitals (rows 35-40) and each year column (H-L):
```
= (Patient_Admissions - Patient_Discharges) / Effective_Bed_Capacity * 100
```
These references should point to the corresponding cells in the three blocks from Step 1. Inspect which block is Admissions, which is Discharges, and which is Bed Capacity by reading the labels in the Task sheet.

**H42:L47 - Column-wise Statistics:**
For each column (H through L), calculate over the six hospital Net Patient Flow values (rows 35:40):
- Row 42: MIN
- Row 43: MAX  
- Row 44: MEDIAN
- Row 45: AVERAGE (simple mean)
- Row 46: 25th percentile
- Row 47: 75th percentile

CRITICAL: Check which row corresponds to which statistic by reading the labels in the Task sheet. Do NOT assume the order above - verify it.

For percentiles, use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid #NAME? errors in some engines. Specifically:
- 25th percentile: `=PERCENTILE(H35:H40, 0.25)`
- 75th percentile: `=PERCENTILE(H35:H40, 0.75)`

Do NOT use `PERCENTILE.INC` or `PERCENTILE.EXC` — these may cause #NAME? errors.

### Step 3: Weighted Mean in H50:L50

For each column H through L:
```
=SUMPRODUCT(<net_patient_flow_range>, <effective_bed_capacity_range>) / SUM(<effective_bed_capacity_range>)
```
Where:
- `<net_patient_flow_range>` = the 6 hospital values in H35:H40 (for column H, etc.)
- `<effective_bed_capacity_range>` = the corresponding Effective Bed Capacity values from H26:H31 (for column H, etc.)

### Saving
1. Create `/root/output/` directory if it doesn't exist.
2. Save the workbook to `/root/output/result.xlsx`.
3. After saving, re-open and verify a few cells contain formula strings (not raw values or errors).

### Key Warnings
- Use `PERCENTILE` not `PERCENTILE.INC`/`PERCENTILE.EXC` to avoid #NAME? errors.
- Write formulas as strings (e.g., `ws['H12'] = '=INDEX(...)'`). Do NOT compute in Python.
- Inspect the actual sheet layout before writing any formulas. Do not assume cell positions.
- Preserve all existing formatting — do not clear or overwrite non-target cells.

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