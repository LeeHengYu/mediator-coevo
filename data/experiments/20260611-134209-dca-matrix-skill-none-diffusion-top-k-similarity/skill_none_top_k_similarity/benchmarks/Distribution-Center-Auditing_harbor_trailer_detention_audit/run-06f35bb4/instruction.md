# Task Instruction

## Task: Harbor Trailer Detention Audit

You must produce two files:
1. `/root/Trailer_Detention_Audit.xlsx`
2. `/root/Trailer_Detention_Brief.docx`

### Step 1: Inspect the source workbook

Read `/root/Trailer_Detention_Log.xlsx` to understand its structure:
- How many sheets does it have? What are the sheet names?
- What are the column headers?
- How many data rows are there?
- Print the first 5 rows and last 2 rows to understand the data.
- Print all unique values of `Seal Required` and `Seal Status` columns (to understand casing).
- Print all unique values of `Carrier` and `Yard` columns.

### Step 2: Build the Excel workbook `/root/Trailer_Detention_Audit.xlsx`

Use `openpyxl` (install if needed: `pip install openpyxl python-docx`). Create the workbook with exactly three sheets named: `RawData`, `Formatted Data`, `Summary`.

#### Sheet 1: `RawData`
- Copy the entire source table exactly as-is (headers + all data rows, preserving original values and order).

#### Sheet 2: `Formatted Data`
- Same row order as RawData.
- First 8 columns exactly as: `Load ID`, `Carrier`, `Allowed Hold Hours`, `Actual Hold Hours`, `Seal Required`, `Seal Status`, `Yard`, `Dispatcher`
  - These must match the source data. If the source columns are in a different order or have different names, map them to these exact header names in this exact order.
- Add 4 new computed columns (columns 9-12) with these exact headers:
  - `Detention Overrun`: 1 if `Actual Hold Hours` > `Allowed Hold Hours`, else 0. Write as integer.
  - `Seal Error`: 1 if `Seal Required` (case-insensitive) == 'YES' AND `Seal Status` (case-insensitive) != 'VERIFIED', else 0. Write as integer.
  - `Total Errors`: `Detention Overrun` + `Seal Error`. Write as integer.
  - `Error Summary`: Exactly one of these strings:
    - `"None"` if Total Errors == 0
    - `"Detention Overrun"` if only detention overrun
    - `"Seal Error"` if only seal error
    - `"Detention Overrun, Seal Error"` if both
- **Write concrete values (integers and strings), NOT Excel formulas.**

#### Sheet 3: `Summary`
- Headers exactly: `Carrier`, `Yard`, `Detention Overrun Errors`, `Seal Errors`, `Total Errors`
- Aggregate from Formatted Data by (Carrier, Yard) groups.
- Include ONLY groups where Total Errors > 0.
- Sort by Carrier ascending (case-sensitive string sort), then Yard ascending.
- Append a final Grand Total row:
  - Carrier = `Grand Total`
  - Yard = `-`
  - Remaining columns = sum of all the rows above in this summary table (i.e., dataset totals for the error columns).

### Step 3: Build the Word document `/root/Trailer_Detention_Brief.docx`

Use `python-docx`. Create `/root/Trailer_Detention_Brief.docx` with:
- A heading (e.g., "Trailer Detention Audit – Executive Summary")
- A short executive summary paragraph (3-6 sentences) that includes ALL of the following:
  1. A plain-language definition of both checks:
     - Detention Overrun: when a trailer's actual hold hours exceed the allowed hold hours.
     - Seal Error: when a trailer requires a seal (Seal Required = YES) but the seal status is not verified.
  2. The computed totals: state the exact number of Detention Overrun errors, Seal errors, and Total Errors found in the dataset.
  3. At least one actionable recommendation (e.g., implement automated alerts, increase dock staffing, review carrier SLAs).
  4. Mention at least two specific carriers that have the highest error counts (identify them from your Summary data — pick the top 2 carriers by total errors).

### Step 4: Validate

After creating both files, run validation:
1. Re-open `/root/Trailer_Detention_Audit.xlsx` and verify:
   - Exactly 3 sheets with exact names: `RawData`, `Formatted Data`, `Summary`
   - `RawData` row count matches source
   - `Formatted Data` has 12 columns with correct headers
   - `Formatted Data` row count matches source
   - Spot-check a few computed values (print first 5 rows of Formatted Data with all 12 columns)
   - `Summary` has correct headers, only groups with Total Errors > 0, is sorted correctly, and has Grand Total row at the end
   - Print the entire Summary sheet
   - Verify Grand Total row sums match the sum of Detention Overrun, Seal Error, and Total Errors from Formatted Data
2. Re-open `/root/Trailer_Detention_Brief.docx` and print all paragraph text to verify content requirements.

### Important Notes
- Sheet names must be EXACTLY `RawData`, `Formatted Data`, `Summary` (watch the space in "Formatted Data").
- Column headers must match EXACTLY as specified (case, spacing, spelling).
- For `Seal Error` logic: compare case-insensitively. `Seal Required` must be 'YES' (any case) AND `Seal Status` must NOT be 'VERIFIED' (any case).
- Write all computed columns as static values, not formulas.
- The Error Summary string must use exactly the specified strings with exact punctuation (`Detention Overrun, Seal Error` — note the comma and single space).
- Do NOT remove the default sheet that openpyxl creates — instead, rename it or delete it so you end up with exactly 3 sheets.

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