# Task Instruction

Build the Excel workbook `/root/Aurora_Rights_Rollforward_4-25.xlsx` by following these steps in order:

## Step 1: Inspect All Input Files

1. Read and print `/root/film_rights_schedule_input.csv` completely.
2. Read and print `/root/music_rights_schedule_input.csv` completely.
3. Read and print `/root/rights_ledger_balances.json` completely.
4. Read and print the three context `.txt` files to understand operational context.
5. Note the exact column headers, row labels, numeric values, date ranges, and any account numbers.

## Step 2: Understand the Required Structure

The workbook follows a rights rollforward / reconciliation pattern:
- **Detail tabs** (`Film Rights #2710` and `Music Rights #2720`): Each has line-item rows starting at row 6, with monthly columns. Column A = labels, columns B onward = months, and column O = totals (sum across months or a key summary column).
- **Control rows** appear below line items in each detail tab: `Month Totals`, `Ending Balance`, `Variance`, `GL Balance`.
- **Summary tab** (`Rights Summary`) aggregates both detail tabs.

Row/column conventions for detail tabs:
- Row 1-5: Headers (title, account info, column headers, etc.)
- Row 6+: Line items (individual rights/licenses)
- Below line items: `Month Totals` (sum of line items), `Ending Balance`, `Variance`, `GL Balance`

Summary tab formula cells:
- B7 = links to Film Rights detail tab column O (likely Ending Balance or a key total)
- B8 = links to Film Rights detail tab column O (another control row)
- B9 = links to Film Rights detail tab column O (another control row)
- B12 = links to Music Rights detail tab column O (parallel to B7)
- B13 = links to Music Rights detail tab column O (parallel to B8)
- B14 = links to Music Rights detail tab column O (parallel to B9)
- B16 = B9 + B14 (combined total across both rights categories)

## Step 3: Build the Workbook with Python + openpyxl

Write a Python script that:

1. **Parses** both CSV files (using `csv` module) and the JSON file (using `json` module). Do NOT modify source files.
2. **Creates** the workbook with exactly 3 sheets in order: `Rights Summary`, `Film Rights #2710`, `Music Rights #2720`.
3. **Populates detail tabs**: Place headers in rows 1-5 (adapt from CSV headers), line items starting at row 6, then control rows (`Month Totals`, `Ending Balance`, `Variance`, `GL Balance`) immediately after the last line item. Use formulas for `Month Totals` (SUM of line item cells in each column). Use values from the JSON for `GL Balance`. `Ending Balance` should be computed (e.g., beginning balance + additions - amortization, or as indicated by input data). `Variance` = `Ending Balance` - `GL Balance`.
4. **Populates summary tab**: Rows 1-6 for headers/labels. B7/B8/B9 reference `='Film Rights #2710'!O<row>` for the appropriate control rows. B12/B13/B14 reference `='Music Rights #2720'!O<row>` similarly. B16 formula = `=B9+B14`.
5. **All numeric values must be stored as numbers**, not strings. Use `float()` or `int()` when writing cell values.
6. **Save** to `/root/Aurora_Rights_Rollforward_4-25.xlsx`.

## Step 4: Validate

1. Reopen the workbook with openpyxl and verify:
   - Exactly 3 sheets with correct names in correct order.
   - Line items start at row 6 in detail tabs.
   - Control row labels (`Month Totals`, `Ending Balance`, `Variance`, `GL Balance`) exist.
   - B7, B8, B9, B12, B13, B14 in Rights Summary contain formulas referencing the detail tabs' column O.
   - B16 contains formula `=B9+B14`.
   - Numeric cells contain numbers, not strings.
2. Print validation results.

## Critical Notes
- Adapt the exact row numbers for control rows based on how many line items exist in each CSV. The control rows go immediately after the last line item.
- When building summary tab references, you must know which row number each control row landed on in each detail tab, then reference column O of those rows.
- If the CSV or JSON structure is ambiguous, print the data and reason about it before coding. Do not guess.
- If openpyxl is not installed, install it with `pip install openpyxl`.

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

Task-local resources are available under `environment/skills`: invoice-organizer, monthly-close.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=noreply@example.com, author_name=Codex Task Generator, category=media-operations, difficulty=medium, tags=[excel, media-rights, invoice-normalization, reconciliation, rollforward].
Verifier config: timeout_sec=900.0.