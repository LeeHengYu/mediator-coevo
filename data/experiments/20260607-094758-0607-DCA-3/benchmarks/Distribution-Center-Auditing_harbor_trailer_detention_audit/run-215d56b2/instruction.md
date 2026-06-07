# Task Instruction

Build two deliverables from `/root/Trailer_Detention_Log.xlsx`:

**Output files (exact paths):**
1. `/root/Trailer_Detention_Audit.xlsx`
2. `/root/Trailer_Detention_Brief.docx`

---

## Step 1: Inspect the source
Open `/root/Trailer_Detention_Log.xlsx` with pandas/openpyxl. Print: sheet name(s), column headers, row count, dtypes, and a few sample rows. Confirm the 8 columns: Load ID, Carrier, Allowed Hold Hours, Actual Hold Hours, Seal Required, Seal Status, Yard, Dispatcher.

## Step 2: Build `/root/Trailer_Detention_Audit.xlsx` with exactly three sheets in this order: `RawData`, `Formatted Data`, `Summary`.

### Sheet `RawData`
- Copy the source table verbatim (same headers, same row order, same values, no extra columns).

### Sheet `Formatted Data`
- First 8 columns identical to RawData in this exact header order:
  1. Load ID
  2. Carrier
  3. Allowed Hold Hours
  4. Actual Hold Hours
  5. Seal Required
  6. Seal Status
  7. Yard
  8. Dispatcher
- Preserve original row order.
- Add columns 9–12 with these exact headers:
  9. `Detention Overrun`
  10. `Seal Error`
  11. `Total Errors`
  12. `Error Summary`
- Compute as concrete written values (no formulas):
  - `Detention Overrun` = 1 if `Actual Hold Hours` > `Allowed Hold Hours` else 0.
  - `Seal Error` = 1 iff `str(Seal Required).strip().upper() == 'YES'` AND `str(Seal Status).strip().upper() != 'VERIFIED'`; else 0.
  - `Total Errors` = `Detention Overrun` + `Seal Error` (integer).
  - `Error Summary` must be exactly one of: `None`, `Detention Overrun`, `Seal Error`, `Detention Overrun, Seal Error` (use this exact casing and comma-space separator).

### Sheet `Summary`
- Headers (exact order):
  1. Carrier
  2. Yard
  3. Detention Overrun Errors
  4. Seal Errors
  5. Total Errors
- Aggregate `Formatted Data` grouped by (Carrier, Yard); sum `Detention Overrun`, `Seal Error`, `Total Errors`.
- Keep only groups where `Total Errors > 0`.
- Sort by Carrier ascending, then Yard ascending.
- Append a final row: Carrier = `Grand Total`, Yard = `-`, remaining columns = column sums across the included groups (which equal the dataset totals of the per-row error flags).
- Write integer values (no formulas).

## Step 3: Build `/root/Trailer_Detention_Brief.docx` (python-docx)
Write a 3–6 sentence executive summary that includes:
- Plain-language definition of `Detention Overrun` (Actual Hold Hours exceeded the Allowed Hold Hours for that load) and `Seal Error` (a load that required a seal did not have its seal status verified).
- Computed totals: total Detention Overrun errors, total Seal errors, and Total Errors (the dataset-wide sums — same as the Grand Total row).
- At least one actionable recommendation (e.g., schedule remediation with worst-performing carriers/yards, enforce seal verification at gate-out).
- Name at least two high-priority carriers with the most exceptions (pick the top carriers by Total Errors from the Summary sheet; list them by name).

## Step 4: Validation before finishing
- Re-open the written xlsx and assert: sheet names == ['RawData','Formatted Data','Summary'] in that order; `Formatted Data` has exactly 12 columns with the exact headers listed; row count of `RawData` == row count of `Formatted Data`.
- Sanity check on a few rows: recompute the four derived values from the first 8 columns and confirm they match what was written.
- Confirm `Error Summary` values are only from the allowed set.
- Confirm the Summary sheet excludes any (Carrier, Yard) group with Total Errors == 0, is sorted correctly, and the Grand Total row's three numeric columns equal the sums of the Formatted Data flag columns.
- Confirm the .docx exists, opens, and contains the totals and at least two carrier names.

## Constraints
- Filenames, sheet names, and column headers must match exactly (including capitalization and spacing).
- Write concrete values, not formulas, in all derived cells.
- Do not add extra sheets, columns, or rows beyond what is specified.

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