# Task Instruction

Execute the following steps in order:

1. **Install required packages**:
 ```
 pip install openpyxl python-docx pandas
 ```

2. **Read the source workbook**:
 - Open `/root/Trailer_Detention_Log.xlsx` using openpyxl or pandas.
 - Inspect the sheet names and the first few rows to identify exact column headers and data layout.
 - Print the column headers and a sample of rows so you understand the mapping.

3. **Create `/root/Trailer_Detention_Audit.xlsx`** with exactly three worksheets: `RawData`, `Formatted Data`, `Summary`.

4. **`RawData` sheet**:
 - Copy the entire source table (headers + all data rows) exactly as-is from the source workbook.

5. **`Formatted Data` sheet**:
 - Keep the same row order as `RawData`.
 - The first 8 columns must have exactly these headers (in order): `Load ID`, `Carrier`, `Allowed Hold Hours`, `Actual Hold Hours`, `Seal Required`, `Seal Status`, `Yard`, `Dispatcher`.
 - Map source columns to these headers. If source headers differ slightly (e.g., spacing, casing), map them correctly but use the exact target header names above.
 - Add 4 new computed columns (columns 9–12) with exactly these headers: `Detention Overrun`, `Seal Error`, `Total Errors`, `Error Summary`.
 - Compute as concrete values (no Excel formulas):
   - `Detention Overrun` = 1 if `Actual Hold Hours` > `Allowed Hold Hours`, else 0.
   - `Seal Error` = 1 if `Seal Required` (case-insensitive) == "YES" AND `Seal Status` (case-insensitive) != "VERIFIED", else 0.
   - `Total Errors` = `Detention Overrun` + `Seal Error`.
   - `Error Summary`: exactly one of `None`, `Detention Overrun`, `Seal Error`, or `Detention Overrun, Seal Error` based on which flags are 1.

6. **`Summary` sheet**:
 - Headers (exactly): `Carrier`, `Yard`, `Detention Overrun Errors`, `Seal Errors`, `Total Errors`.
 - Group from `Formatted Data` by (Carrier, Yard).
 - Sum `Detention Overrun` → `Detention Overrun Errors`, sum `Seal Error` → `Seal Errors`, sum `Total Errors` → `Total Errors` for each group.
 - Include only groups where `Total Errors > 0`.
 - Sort by `Carrier` ascending, then `Yard` ascending.
 - Append a final row: `Carrier` = `Grand Total`, `Yard` = `-`, and the remaining columns are the dataset-wide totals of those three error columns.

7. **Create `/root/Trailer_Detention_Brief.docx`**:
 - Write a short executive summary (3–6 sentences) that:
     a. Defines both checks in plain language: Detention Overrun means actual hold hours exceeded the allowed threshold; Seal Error means a seal was required but not verified.
     b. States the computed totals for Detention Overrun errors, Seal errors, and Total Errors (use the Grand Total row values).
     c. Identifies at least two carriers with the highest total errors by name.
     d. Includes at least one actionable recommendation (e.g., tighter dispatch protocols, seal verification audits).

8. **Validation**:
 - Re-open `/root/Trailer_Detention_Audit.xlsx` and verify:
     - Sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
     - `Formatted Data` has exactly 12 columns with the correct headers.
     - `Summary` last row has `Carrier` == `Grand Total` and `Yard` == `-`.
     - All computed columns contain concrete numeric/string values, not formulas.
   - Re-open `/root/Trailer_Detention_Brief.docx` and print its text to confirm it meets all content requirements.
   - Print confirmation of all checks passing.

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