# Task Instruction

Create a Python script `/root/solve.py` and execute it to produce the two deliverables. The script should do the following:

1. **Read the source workbook**:
   - `import openpyxl, pandas as pd, docx` (install if needed: `pip install openpyxl python-docx pandas`)
   - Read `/root/Trailer_Detention_Log.xlsx` into a DataFrame.

2. **Create `/root/Trailer_Detention_Audit.xlsx`** with exactly three sheets: `RawData`, `Formatted Data`, `Summary`.

3. **`RawData` sheet**:
   - Copy the source table exactly as-is (same columns, same row order, same values).

4. **`Formatted Data` sheet**:
   - Keep the same row order as RawData.
   - Keep the first 8 columns exactly as: `Load ID`, `Carrier`, `Allowed Hold Hours`, `Actual Hold Hours`, `Seal Required`, `Seal Status`, `Yard`, `Dispatcher`.
   - Map source column names to these target names case-insensitively (e.g., if source has slightly different casing or spacing, map accordingly). Print the source columns for debugging.
   - Add four computed columns (as concrete values, NOT formulas):
     - `Detention Overrun`: 1 if `Actual Hold Hours` > `Allowed Hold Hours`, else 0. Store as int.
     - `Seal Error`: 1 if `Seal Required` upper-stripped == 'YES' AND `Seal Status` upper-stripped != 'VERIFIED', else 0. Store as int.
     - `Total Errors`: `Detention Overrun` + `Seal Error`. Store as int.
     - `Error Summary`: exactly one of `'None'`, `'Detention Overrun'`, `'Seal Error'`, `'Detention Overrun, Seal Error'` based on which flags are 1.

5. **`Summary` sheet**:
   - Group `Formatted Data` by `(Carrier, Yard)`.
   - For each group, sum `Detention Overrun` → `Detention Overrun Errors`, sum `Seal Error` → `Seal Errors`, sum `Total Errors` → `Total Errors`.
   - Filter to only groups where `Total Errors > 0`.
   - Sort by `Carrier` ascending then `Yard` ascending.
   - Headers must be exactly: `Carrier`, `Yard`, `Detention Overrun Errors`, `Seal Errors`, `Total Errors`.
   - Append a final row: `Carrier`=`Grand Total`, `Yard`=`-`, and the remaining columns are the dataset-wide sums of the respective error columns.

6. **Write the Excel file** using `openpyxl` engine via pandas `ExcelWriter` (or openpyxl directly). Ensure sheet names are exactly `RawData`, `Formatted Data`, `Summary`.

7. **Create `/root/Trailer_Detention_Brief.docx`**:
   - Use `python-docx` to create a Word document.
   - Add a heading: `Trailer Detention Audit – Executive Summary`.
   - Write a 3-6 sentence executive summary paragraph that includes:
     - Plain-language definition of both checks: "A Detention Overrun occurs when a trailer's actual hold hours exceed the allowed hold hours. A Seal Error occurs when a seal is required but its status is not verified."
     - The computed totals: total Detention Overrun errors, total Seal errors, total combined errors (use the actual numbers from the data).
     - Identify the top 2 carriers by total errors and mention them by name as high-priority.
     - At least one actionable recommendation (e.g., "We recommend implementing automated hold-time alerts and mandatory seal verification checklists for these carriers.").

8. **Validation steps** (print to stdout):
   - Re-open `/root/Trailer_Detention_Audit.xlsx` and print sheet names.
   - Print the first 3 rows and last row of `Formatted Data`.
   - Print all rows of `Summary`.
   - Print the column headers of each sheet.
   - Confirm `/root/Trailer_Detention_Brief.docx` exists and print its paragraph text.

Execute: `cd /root && pip install openpyxl python-docx pandas && python solve.py`

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