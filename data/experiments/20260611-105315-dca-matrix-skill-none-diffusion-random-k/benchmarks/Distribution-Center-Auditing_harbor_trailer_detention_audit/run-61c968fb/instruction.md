# Task Instruction

## Task: Harbor Trailer Detention Audit

You must produce two files:
1. `/root/Trailer_Detention_Audit.xlsx`
2. `/root/Trailer_Detention_Brief.docx`

### Step 1: Inspect the source workbook

```bash
pip install openpyxl python-docx pandas
```

Then read `/root/Trailer_Detention_Log.xlsx` to understand its structure:
- How many rows/columns?
- What are the exact column headers?
- What values appear in `Seal Required` and `Seal Status` columns?
- Print the first 5 and last 5 rows.

### Step 2: Build the Excel workbook `/root/Trailer_Detention_Audit.xlsx`

Write a Python script that does the following:

#### Sheet 1: `RawData`
- Copy the entire source table from `Trailer_Detention_Log.xlsx` exactly as-is (same headers, same data, same row order).

#### Sheet 2: `Formatted Data`
- Same row order as RawData.
- First 8 columns must be exactly (use these exact header strings):
  1. `Load ID`
  2. `Carrier`
  3. `Allowed Hold Hours`
  4. `Actual Hold Hours`
  5. `Seal Required`
  6. `Seal Status`
  7. `Yard`
  8. `Dispatcher`
- Map the source columns to these 8 columns. If the source uses slightly different names, map them correctly based on inspection.
- Add 4 new computed columns (columns 9–12) with these EXACT headers:
  9. `Detention Overrun`
  10. `Seal Error`
  11. `Total Errors`
  12. `Error Summary`

Computation rules (write concrete values, NOT Excel formulas):
- `Detention Overrun` = 1 if `Actual Hold Hours` > `Allowed Hold Hours`, else 0
- `Seal Error` = 1 if (`Seal Required` upper == `YES`) AND (`Seal Status` upper != `VERIFIED`), else 0
- `Total Errors` = `Detention Overrun` + `Seal Error`
- `Error Summary` = exactly one of these strings:
  - `None` (if Total Errors == 0)
  - `Detention Overrun` (if only detention overrun)
  - `Seal Error` (if only seal error)
  - `Detention Overrun, Seal Error` (if both)

#### Sheet 3: `Summary`
- Headers exactly: `Carrier`, `Yard`, `Detention Overrun Errors`, `Seal Errors`, `Total Errors`
- Group `Formatted Data` by (Carrier, Yard).
- Sum `Detention Overrun` → `Detention Overrun Errors`, sum `Seal Error` → `Seal Errors`, sum `Total Errors` → `Total Errors`.
- Include ONLY groups where `Total Errors > 0`.
- Sort by `Carrier` ascending, then `Yard` ascending (standard alphabetical).
- Append a final row: `Carrier` = `Grand Total`, `Yard` = `-`, and the remaining three columns = the sum of all rows above (i.e., dataset-wide totals).

### Step 3: Build the Word document `/root/Trailer_Detention_Brief.docx`

Create `/root/Trailer_Detention_Brief.docx` containing an executive summary paragraph (3–6 sentences) that includes ALL of the following:
1. A plain-language definition of both checks:
   - Detention Overrun: when a trailer's actual hold time exceeds the allowed hold hours.
   - Seal Error: when a trailer requires a seal (Seal Required = YES) but the seal status is not verified.
2. The exact computed totals: state the number of Detention Overrun errors, Seal errors, and Total Errors from the Grand Total row.
3. At least one actionable recommendation (e.g., implement automated alerts, increase dock staffing, review carrier SLAs).
4. Mention at least two specific carrier names that have the highest error counts (look at the Summary sheet to identify them).

### Step 4: Validate

After creating both files, run validation:
1. Re-open `/root/Trailer_Detention_Audit.xlsx` and verify:
   - Sheet names are exactly `RawData`, `Formatted Data`, `Summary` (check spelling and case).
   - `RawData` row count matches source.
   - `Formatted Data` has 12 columns with exact header names as specified.
   - `Formatted Data` row count matches source.
   - `Summary` has 5 columns with exact header names.
   - `Summary` last row has `Carrier` = `Grand Total` and `Yard` = `-`.
   - Grand Total row totals match the sums from Formatted Data.
   - All `Error Summary` values are one of the four allowed strings.
2. Re-open `/root/Trailer_Detention_Brief.docx` and print its text to confirm it contains the required elements.

Print all validation results clearly. Fix any issues before finishing.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=medium, tags=[excel, openpyxl, docx, audit, logistics].
Verifier config: timeout_sec=900.0.