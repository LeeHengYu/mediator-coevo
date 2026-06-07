# Task Instruction

Build two deliverables from `/root/Receiving_Log.xlsx`:

## Step 1: Inspect the source
- Open `/root/Receiving_Log.xlsx` with pandas/openpyxl. Identify the sheet name and confirm columns include: Receipt ID, Item Code, Expected Qty, Received Qty, Storage Class, Temp Status, Supplier, Dock.
- Print head/dtypes to confirm column names match exactly (watch for whitespace/case).

## Step 2: Create `/root/Receiving_Exception_Audit.xlsx` with EXACTLY three sheets in this order: `RawData`, `Formatted Data`, `Summary`.

### Sheet `RawData`
- Write the source dataframe exactly as read (same rows, same column order/names as source).

### Sheet `Formatted Data`
- Preserve row order from RawData.
- Columns 1–8 (exact headers, in this order): `Receipt ID`, `Item Code`, `Expected Qty`, `Received Qty`, `Storage Class`, `Temp Status`, `Supplier`, `Dock`.
- Add columns 9–12 with these exact headers: `Qty Variance`, `Cold Chain Error`, `Total Errors`, `Error Summary`.
- Compute concrete numeric/text values in Python (NOT spreadsheet formulas):
  - `Qty Variance` = 1 if `Received Qty` != `Expected Qty` else 0.
  - `Cold Chain Error` = 1 if `Storage Class` (case-insensitive strip) in {`CHILLED`, `FROZEN`} AND `Temp Status` (case-insensitive strip) != `OK`; else 0.
  - `Total Errors` = `Qty Variance` + `Cold Chain Error`.
  - `Error Summary` mapping:
    - (0,0) -> `None`
    - (1,0) -> `Qty Variance`
    - (0,1) -> `Cold Chain Error`
    - (1,1) -> `Qty Variance, Cold Chain Error`

### Sheet `Summary`
- Headers (exact, in order): `Item Code`, `Supplier`, `Qty Variance Errors`, `Cold Chain Errors`, `Total Errors`.
- Aggregate from `Formatted Data` grouped by (`Item Code`, `Supplier`): sum of `Qty Variance`, sum of `Cold Chain Error`, sum of `Total Errors`.
- Keep ONLY groups where `Total Errors > 0`.
- Sort by `Item Code` ascending, then `Supplier` ascending.
- Append a final Grand Total row: `Item Code`=`Grand Total`, `Supplier`=`-`, then dataset-wide sums of the three error columns (sums over the full Formatted Data, equivalently sum of the filtered/non-filtered groups — use full dataset totals to be safe; since groups with 0 contribute 0 either way, group sums equal dataset totals).

## Step 3: Create `/root/Receiving_Exception_Brief.docx`
Use `python-docx`. Write a 3–6 sentence executive summary that MUST include all of:
1. Plain-language definitions of both checks:
   - `Qty Variance`: a receipt whose Received Qty does not match Expected Qty.
   - `Cold Chain Error`: a CHILLED or FROZEN item whose Temp Status is not OK upon receipt.
2. Computed totals (use the exact integer counts from your data): total `Qty Variance` errors = X, total `Cold Chain` errors = Y, total errors = Z. Write the numerals explicitly.
3. At least one concrete actionable recommendation (e.g., audit specific supplier dock handoffs, retrain receivers on temperature logging, escalate to QA).
4. Mention at least TWO high-priority item codes. To select them: sort the Summary (excluding Grand Total) by `Total Errors` descending and pick the top 2 distinct `Item Code` values. Write the literal item code strings verbatim into the docx text (e.g., "Top exception items include ITEM-123 and ITEM-456."). Ensure the item codes appear as plain text in the document.

## Step 4: Validation before finishing
- Reopen `/root/Receiving_Exception_Audit.xlsx`; verify sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
- Verify `Formatted Data` has 12 columns with the exact headers listed above.
- Verify Summary headers and that the Grand Total row exists with `Supplier`=`-`.
- Reopen the docx and confirm: both check definitions present, the three numeric totals present as digits, at least one recommendation, and the two chosen item codes appear as substrings.

Do not use spreadsheet formulas; write concrete values. Do not alter filenames or sheet names.

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