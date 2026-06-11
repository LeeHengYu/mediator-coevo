# Task Instruction

Create a Python script `/root/solve.py` and execute it to produce `/root/Trailer_Detention_Audit.xlsx` and `/root/Trailer_Detention_Brief.docx`.

## Step-by-step instructions

### 1. Inspect the source file
```python
import pandas as pd
df = pd.read_excel('/root/Trailer_Detention_Log.xlsx')
print(df.columns.tolist())
print(df.head(10))
print(df.dtypes)
print(df.isnull().sum())
```
Run this first to understand exact column names, data types, and where NaN values appear. Note the exact column names from the source — they may differ slightly from the spec names.

### 2. Write and run `/root/solve.py`

The script must do the following:

#### A. Read source data
- Read `/root/Trailer_Detention_Log.xlsx` into a pandas DataFrame.
- **Critical**: After reading, replace ALL NaN/None values with the string `'N/A'` using `df.fillna('N/A')`. This is essential — the verifier expects `'N/A'` strings, not None/NaN, for empty cells. Do this immediately after reading, before any processing.
- Verify the first 8 columns match: Load ID, Carrier, Allowed Hold Hours, Actual Hold Hours, Seal Required, Seal Status, Yard, Dispatcher. If column names differ slightly from the spec, rename them to match exactly.

#### B. Create the Excel workbook with openpyxl
Use `openpyxl` to create `/root/Trailer_Detention_Audit.xlsx` with exactly three sheets: `RawData`, `Formatted Data`, `Summary`.

#### C. `RawData` sheet
- Write the source data exactly as read (after NaN→'N/A' replacement), preserving all original columns, row order, and values.
- Include the header row.
- Ensure numeric columns (like Allowed Hold Hours, Actual Hold Hours) remain as numbers (int/float), not strings. Only truly empty cells should become 'N/A'.

#### D. `Formatted Data` sheet
- Same rows and order as RawData.
- First 8 columns exactly as specified: Load ID, Carrier, Allowed Hold Hours, Actual Hold Hours, Seal Required, Seal Status, Yard, Dispatcher.
- Add 4 computed columns (9-12) with these exact headers: `Detention Overrun`, `Seal Error`, `Total Errors`, `Error Summary`.
- Computation rules:
  - `Detention Overrun`: 1 if `Actual Hold Hours > Allowed Hold Hours`, else 0. Both must be numeric for comparison. If either is 'N/A', treat as 0.
  - `Seal Error`: 1 if `Seal Required` (case-insensitive) == 'YES' AND `Seal Status` (case-insensitive) != 'VERIFIED', else 0. Handle 'N/A' Seal Status: if Seal Required is YES and Seal Status is 'N/A', that IS a seal error (since 'N/A' != 'VERIFIED').
  - `Total Errors` = `Detention Overrun` + `Seal Error` (write as integer).
  - `Error Summary`: exactly one of `'None'`, `'Detention Overrun'`, `'Seal Error'`, `'Detention Overrun, Seal Error'` based on which flags are 1.
- Write concrete values (integers and strings), NOT Excel formulas.

#### E. `Summary` sheet
- Headers: `Carrier`, `Yard`, `Detention Overrun Errors`, `Seal Errors`, `Total Errors`.
- Group `Formatted Data` by (Carrier, Yard).
- Sum `Detention Overrun` → `Detention Overrun Errors`, sum `Seal Error` → `Seal Errors`, sum `Total Errors` → `Total Errors`.
- Include only groups where Total Errors > 0.
- Sort by Carrier ascending, then Yard ascending.
- Append a Grand Total row: Carrier=`Grand Total`, Yard=`-`, and sums of the three error columns across all included groups.

#### F. Save the Excel file
- Save to `/root/Trailer_Detention_Audit.xlsx`.
- Ensure no extra default sheets exist (remove 'Sheet' if openpyxl creates one).

#### G. Create `/root/Trailer_Detention_Brief.docx`
- Use `python-docx`.
- Write a 3-6 sentence executive summary paragraph that includes:
  1. Plain-language definition of Detention Overrun check (actual hold hours exceeding allowed hold hours).
  2. Plain-language definition of Seal Error check (seal required but not verified).
  3. The exact computed totals: X Detention Overrun errors, Y Seal errors, Z Total Errors.
  4. At least one actionable recommendation (e.g., implement automated alerts, review carrier SLAs).
  5. Mention at least two carriers with the highest error counts by name.

### 3. Validation after running
After running the script:
1. Re-read `/root/Trailer_Detention_Audit.xlsx` and verify:
   - Sheet names are exactly `['RawData', 'Formatted Data', 'Summary']`.
   - RawData has no None/NaN values (all should be 'N/A').
   - Formatted Data has 12 columns with correct headers.
   - Summary has the Grand Total row as the last row.
2. Re-read `/root/Trailer_Detention_Brief.docx` and verify it contains the required elements.
3. Print a few sample rows from each sheet to confirm correctness.

### Critical reminders
- The NaN→'N/A' replacement is the #1 priority fix from previous failure. Do it immediately after `pd.read_excel()` and before ANY other processing.
- When writing to openpyxl, ensure that numeric values stay numeric (int) and string values stay as strings. Do NOT convert numbers to strings.
- For the NaN replacement: use `df = df.fillna('N/A')` but then ensure numeric columns that had no NaN remain as int/float. A safer approach: identify which cells are NaN before fillna, or use `df = df.where(df.notna(), 'N/A')` and then cast numeric columns back to their proper types.
- Install any needed packages: `pip install openpyxl python-docx` if not already available.

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