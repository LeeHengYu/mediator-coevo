# Task Instruction

Build two deliverables from `/root/Receiving_Log.xlsx`:

**Output files (exact paths):**
1. `/root/Receiving_Exception_Audit.xlsx`
2. `/root/Receiving_Exception_Brief.docx`

## Step 1: Inspect source
Load `/root/Receiving_Log.xlsx` with pandas/openpyxl. Confirm the columns include: Receipt ID, Item Code, Expected Qty, Received Qty, Storage Class, Temp Status, Supplier, Dock. Print head and dtypes before proceeding.

## Step 2: Build `/root/Receiving_Exception_Audit.xlsx` with EXACTLY three sheets in this order: `RawData`, `Formatted Data`, `Summary`.

### Sheet `RawData`
- Write the source table exactly as-is (same columns, same row order, same values).

### Sheet `Formatted Data`
- Same row order as RawData.
- First 8 columns exactly (in this order, with these exact headers):
  1. Receipt ID
  2. Item Code
  3. Expected Qty
  4. Received Qty
  5. Storage Class
  6. Temp Status
  7. Supplier
  8. Dock
- Append columns 9-12 with these exact headers:
  9. `Qty Variance`
  10. `Cold Chain Error`
  11. `Total Errors`
  12. `Error Summary`
- Computation rules (write concrete values, not formulas):
  - `Qty Variance` = 1 if `Received Qty` != `Expected Qty`, else 0.
  - `Cold Chain Error` = 1 only if `Storage Class` (case-insensitive) is `CHILLED` or `FROZEN` AND `Temp Status` (case-insensitive) is not `OK`. Else 0. Use `.str.upper().str.strip()` for the comparison but do NOT mutate the original column values written to the sheet.
  - `Total Errors` = `Qty Variance` + `Cold Chain Error` (integer).
  - `Error Summary` must be exactly one of: `None`, `Qty Variance`, `Cold Chain Error`, `Qty Variance, Cold Chain Error` (note the comma-space).

### Sheet `Summary`
- Headers exactly: `Item Code`, `Supplier`, `Qty Variance Errors`, `Cold Chain Errors`, `Total Errors`.
- Group `Formatted Data` by (`Item Code`, `Supplier`), sum `Qty Variance` -> `Qty Variance Errors`, sum `Cold Chain Error` -> `Cold Chain Errors`, sum `Total Errors` -> `Total Errors`.
- Include only groups where `Total Errors > 0`.
- Sort by `Item Code` ascending, then `Supplier` ascending.
- Append a final row literally: `Item Code` = `Grand Total`, `Supplier` = `-`, then the three numeric column totals across the included groups (these equal the dataset totals of the three error columns).

Use `pandas.ExcelWriter(..., engine='openpyxl')` and write each frame with `index=False`.

## Step 3: Build `/root/Receiving_Exception_Brief.docx`
Use python-docx. Write an executive summary of 3-6 sentences that contains ALL of:
- Plain-language definition of `Qty Variance` (received qty differs from expected qty).
- Plain-language definition of `Cold Chain Error` (chilled/frozen item arrived with temp status not OK).
- The computed totals: total Qty Variance errors, total Cold Chain errors, total Total Errors (use the dataset totals from Formatted Data).
- At least one actionable recommendation (e.g., audit specific suppliers/docks, reinforce cold-chain handoff).
- Mention at least two specific high-priority Item Codes with the most exceptions (pick the top 2 by `Total Errors` from the Summary sheet, ties broken by item code).

Save the docx.

## Step 4: Validate before finishing
- Reopen `/root/Receiving_Exception_Audit.xlsx`; assert sheet names equal exactly `['RawData', 'Formatted Data', 'Summary']`.
- Assert `Formatted Data` has 12 columns with the exact headers listed above.
- Assert every `Error Summary` value is in the allowed 4-value set.
- Assert `Total Errors` equals `Qty Variance + Cold Chain Error` on every row.
- Assert Summary contains only rows with Total Errors > 0 and ends with a `Grand Total` / `-` row whose numeric values equal the column sums of the preceding rows.
- Reopen the docx and confirm it contains the two definitions, three totals (as numbers), a recommendation, and two item codes.

Report any validation failure and fix before exiting.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=medium, tags=[excel, openpyxl, docx, audit, warehouse].
Verifier config: timeout_sec=900.0.