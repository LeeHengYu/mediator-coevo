# Task Instruction

Execute the following steps in order to produce the two deliverables.

## 1. Inspect the source workbook
```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('/root/Trailer_Detention_Log.xlsx')
for s in wb.sheetnames:
    ws = wb[s]
    print(f'Sheet: {s}  rows={ws.max_row}  cols={ws.max_column}')
    for r in ws.iter_rows(min_row=1, max_row=min(5, ws.max_row), values_only=True):
        print(r)
"
```
Note the exact header names and their column positions. Map them to the 8 required target headers:
1. Load ID
2. Carrier
3. Allowed Hold Hours
4. Actual Hold Hours
5. Seal Required
6. Seal Status
7. Yard
8. Dispatcher

## 2. Build `/root/Trailer_Detention_Audit.xlsx`

Write a single Python script that:

### a) Read source data
- Use openpyxl to read all data rows from the first (or only) sheet of `Trailer_Detention_Log.xlsx`.
- Build a list of dicts, normalizing source header names to the 8 target names (strip whitespace, handle casing differences).

### b) Sheet `RawData`
- Create a new workbook. Rename the default sheet to `RawData`.
- Write the source headers in row 1 exactly as they appear in the source file.
- Copy every data row exactly as-is.

### c) Sheet `Formatted Data`
- Create sheet named exactly `Formatted Data`.
- Row 1 headers (12 columns): Load ID, Carrier, Allowed Hold Hours, Actual Hold Hours, Seal Required, Seal Status, Yard, Dispatcher, Detention Overrun, Seal Error, Total Errors, Error Summary.
- For each data row (same order as RawData):
  - Columns 1-8: copy from source.
  - `Detention Overrun` = 1 if float(Actual Hold Hours) > float(Allowed Hold Hours), else 0.
  - `Seal Error` = 1 if str(Seal Required).strip().upper() == 'YES' and str(Seal Status).strip().upper() != 'VERIFIED', else 0.
  - `Total Errors` = Detention Overrun + Seal Error.
  - `Error Summary`: build from parts — if both flags: 'Detention Overrun, Seal Error'; if only detention: 'Detention Overrun'; if only seal: 'Seal Error'; else 'None'.
- Write concrete values (int for flags, str for summary). Do NOT use Excel formulas.

### d) Sheet `Summary`
- Create sheet named exactly `Summary`.
- Headers: Carrier, Yard, Detention Overrun Errors, Seal Errors, Total Errors.
- Group rows from `Formatted Data` by (Carrier, Yard). For each group compute sum of Detention Overrun, sum of Seal Error, sum of Total Errors.
- Keep only groups where Total Errors > 0.
- Sort by Carrier ascending then Yard ascending (standard string sort).
- Append a final row: Carrier='Grand Total', Yard='-', and the three columns are the dataset-wide sums of Detention Overrun Errors, Seal Errors, Total Errors across ALL formatted data rows (not just the filtered groups — though they should match since groups with 0 contribute nothing).
- Save the workbook to `/root/Trailer_Detention_Audit.xlsx`.

## 3. Build `/root/Trailer_Detention_Brief.docx`

Using python-docx, create a Word document at `/root/Trailer_Detention_Brief.docx` containing:
- A heading (e.g., 'Trailer Detention Audit – Executive Summary').
- A paragraph of 3-6 sentences that includes ALL of the following:
  1. A plain-language definition of both checks: explain that a Detention Overrun occurs when a trailer's actual hold time exceeds the allowed hold hours, and a Seal Error occurs when a seal is required but not verified.
  2. The computed totals: state the exact number of Detention Overrun errors, Seal errors, and Total Errors found in the dataset.
  3. At least one actionable recommendation (e.g., implement automated alerts, increase dock staffing, enforce seal verification protocols).
  4. Mention at least two carriers by name that have the highest error counts (pick the top 2 carriers by total errors from the Summary data).

## 4. Verify outputs
- Confirm `/root/Trailer_Detention_Audit.xlsx` exists and has sheets named exactly `RawData`, `Formatted Data`, `Summary`.
- Print a few rows from each sheet.
- Confirm `/root/Trailer_Detention_Brief.docx` exists and print its paragraph text.
- Check that the Grand Total row values are consistent with the Formatted Data totals.

Write and run the complete Python script. If any step fails, diagnose and fix before proceeding.

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