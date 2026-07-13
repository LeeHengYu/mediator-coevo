# Task Instruction

Execute the following steps inside a single Python script `/app/workspace/solve.py`, then run it.

## Step 0 – Reconnaissance

```bash
find /app/workspace/dataset -type f | head -40
```
List all image files recursively under `/app/workspace/dataset` to understand the folder structure and file types.

## Step 1 – Write and run `solve.py`

Create `/app/workspace/solve.py` with the logic below.

### 1.1 Collect image paths
- Recursively walk `/app/workspace/dataset`.
- Collect every file whose extension (case-insensitive) is one of: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.gif`, `.webp`.
- For each image compute:
  - `relative_path`: path relative to `/app/workspace/dataset`, always using forward slashes.
  - `batch_name`: the first directory component of that relative path (i.e., the top-level folder directly under `dataset`).

### 1.2 OCR each image
- Use `pytesseract.image_to_string` (with Pillow to open the image).
- Optionally try `--psm 6` if default gives poor results, but start with default.
- Store the full OCR text for each image.

### 1.3 Classify – keep only fuel receipts
An image is a fuel receipt **only if** its OCR text (case-insensitive) contains at least one of:
- `FUEL RECEIPT`
- `PUMP SALE`
- `TAX INVOICE`

Discard all other images entirely.

### 1.4 Extract fields from each fuel receipt

#### Transaction reference (`txn_ref`)
Search the OCR text (case-insensitive) for lines matching patterns like:
- `Txn Ref`, `Transaction No`, `Ref No`
The reference value is typically on the same line after a colon/space, or on the next line. Extract the alphanumeric reference string. Strip surrounding whitespace.

#### Date (`date`)
Search for labels: `Sale Date`, `Date`.
Extract the date string. Parse it:
- Try `DD/MM/YYYY` first (prefer this for ambiguous dates).
- Then `DD-MM-YYYY`.
- Then `MM/DD/YYYY`.
- Then `YYYY-MM-DD`.
Output in ISO `YYYY-MM-DD` format.

Be careful: when the day value is ≤ 12 and month value is also ≤ 12, prefer DD/MM/YYYY interpretation.

#### Total amount (`total_amount`)
Search for lines containing (case-insensitive): `GRAND TOTAL`, `TOTAL AMOUNT`, `AMOUNT PAID`, or `TOTAL`.
**Ignore** any line also containing: `DISCOUNT`, `CASHBACK`, `SAVINGS`, `LOYALTY`, `TAX` (case-insensitive).
From the matching line (or the next line if the number isn't on the same line), extract a numeric value. Use a regex like `[\d,]+\.\d{2}` or `[\d]+\.\d{2}`. Remove commas. If multiple candidate lines match, prefer `GRAND TOTAL` > `TOTAL AMOUNT` > `AMOUNT PAID` > `TOTAL` (most specific first). Format as a string with exactly two decimal places (e.g., `"45.30"`).

If a field cannot be extracted, still include the row but use empty string for that field (this should be rare).

### 1.5 Build the output DataFrame
- Columns in order: `batch_name`, `relative_path`, `txn_ref`, `date`, `total_amount`.
- Sort by `relative_path` ascending.
- Deduplicate: if the same `txn_ref` appears more than once, keep only the first occurrence (by `relative_path` order). Drop later duplicates.
- Only non-empty `txn_ref` values should be considered for deduplication (if `txn_ref` is empty, keep the row).

### 1.6 Write to Excel
- Write to `/app/workspace/fuel_packets.xlsx` using `openpyxl`.
- Sheet name must be exactly `transactions`.
- Write header row, then data rows. No extra sheets, rows, or columns.
- Ensure `total_amount` is written as a string, not a number (to preserve two decimal places). You can set the cell's `number_format` to `'@'` (text) or simply write string values.

## Step 2 – Run
```bash
cd /app/workspace && python solve.py
```

## Step 3 – Validate
After running, verify:
1. The file `/app/workspace/fuel_packets.xlsx` exists.
2. Open it with openpyxl and confirm:
   - Sheet name is `transactions`.
   - Column headers are exactly `['batch_name', 'relative_path', 'txn_ref', 'date', 'total_amount']`.
   - All rows have non-empty `txn_ref`, `date`, and `total_amount` (warn if any are empty).
   - `date` values match `YYYY-MM-DD` pattern.
   - `total_amount` values have exactly two decimal places.
   - Rows are sorted by `relative_path`.
   - No duplicate `txn_ref` values.
   - Print the number of rows and first few rows for inspection.

Run a small validation script or inline check after `solve.py` completes.

## Important implementation notes
- When extracting the total amount, be very careful about OCR artifacts. Dollar signs, spaces, commas may appear. Strip `$` and `,` before parsing.
- Some receipts may have the keyword on one line and the number on the next line. Handle this by checking the next line if no number is found on the keyword line.
- For date parsing, use `datetime.strptime` and catch exceptions to try multiple formats.
- Write `total_amount` as a Python string to the cell so openpyxl doesn't convert it to float.
- Use `os.path.relpath` and replace `\\` with `/` for `relative_path`.
- Print progress/debug info (image path, classification result, extracted fields) to stdout so we can diagnose issues if needed.

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
Task metadata: author_email=benchmark@example.com, author_name=Benchmark Designer, category=data extraction, difficulty=hard, tags=[image ocr, fuel receipts, nested packets].
Verifier config: env={}, timeout_sec=600.0.