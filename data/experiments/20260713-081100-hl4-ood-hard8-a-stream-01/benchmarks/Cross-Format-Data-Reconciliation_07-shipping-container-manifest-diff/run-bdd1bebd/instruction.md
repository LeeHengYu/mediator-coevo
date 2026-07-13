# Task Instruction

## Task: Reconcile Shipping Container Manifest (PDF vs Excel)

You need to compare an archived PDF snapshot against a current Excel workbook and produce a JSON diff report.

### Step-by-step Instructions

**Step 1: Inspect the input files**
- List `/root/` to confirm both files exist: `container_manifest_old.pdf` and `container_manifest_current.xlsx`.
- Open and inspect the Excel file first to understand the column names and data types. Print the first few rows and the shape.
- Note the exact column names (especially the ID column and any numeric columns like `WeightTons`).

**Step 2: Extract the table from the PDF**
- Install any needed libraries: `pip install tabula-py camelot-py[cv] pdfplumber openpyxl pandas`
- Try `pdfplumber` first to extract the table from the PDF. Read all pages.
- Print the first few rows of the extracted PDF table and its shape to verify extraction worked.
- If `pdfplumber` extraction is poor (empty or garbled), fall back to `tabula-py` or `camelot`.
- **Critical**: After extraction, verify the PDF dataframe has the same column names as the Excel file. If column names differ, align them. Print both sets of column names for comparison.
- Verify that ID values match the pattern `^CNT\d{4}$`. Print a count of valid IDs from each source.

**Step 3: Clean and align data types**
- Strip whitespace from all string columns in both dataframes.
- For the ID column, ensure consistent formatting (string type, no extra spaces).
- For numeric columns (like `WeightTons`), convert to float in both dataframes. Handle any parsing issues.
- For text/string columns, ensure they are strings (not NaN — if a value is NaN in one but not the other, that's a change).

**Step 4: Compute the diff**
- **Missing containers**: Find IDs present in the PDF (old) but absent in the Excel (current). Collect these as a sorted list of ID strings.
- **Changed containers**: For IDs present in both, compare each field (excluding the ID column itself). For each field where the old value differs from the new value:
  - Emit an object with `id`, `field`, `old_value`, `new_value`.
  - Numeric fields must be output as numbers (float or int as appropriate). Use `round()` if needed to avoid floating point artifacts, but only round to the precision present in the source data.
  - Text fields must be output as strings.
  - If a record has multiple changed fields, emit one object per field.
- Sort `changed_containers` by `id` first, then by `field` for deterministic output.
- **Important**: Do NOT include records that are only in the new file (added containers) — only missing and changed.

**Step 5: Build and write the JSON report**
- Construct the final dict:
```python
{
  "missing_containers": [...],  # sorted list of ID strings
  "changed_containers": [...]   # sorted list of change objects
}
```
- Before writing, print the number of missing containers and the number of changed container entries to sanity-check.
- Print a sample of the output (first 5 missing, first 10 changes) for visual verification.
- Write to `/root/container_diff_report.json` using `json.dump` with `indent=2`.
- **After writing**, re-read the file and parse it with `json.load` to verify it's valid JSON. Print confirmation.

**Step 6: Final validation**
- Verify the JSON file exists and is non-empty.
- Verify `missing_containers` is a list of strings matching `^CNT\d{4}$`.
- Verify `changed_containers` entries each have exactly the keys: `id`, `field`, `old_value`, `new_value`.
- Verify numeric `old_value`/`new_value` are actual numbers (not strings) for numeric fields.
- Verify sorting is correct.

### Key Pitfalls to Avoid
- PDF table extraction can be unreliable. Always print and verify extracted data before proceeding.
- Floating point comparison: compare values after rounding to reasonable precision (e.g., 1-2 decimal places for weights) to avoid false positives from float representation.
- Don't confuse NaN with actual missing data — if both old and new are NaN for a field, that's not a change.
- Make sure the JSON output uses Python native types (not numpy types) — convert with `.item()` or explicit casting before serialization.

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
Task metadata: author_email=task-engineer@meituan.com, author_name=CatPaw Task Engineer, category=logistics, difficulty=hard, tags=[pdf, xlsx, shipping, manifest, diff].
Verifier config: timeout_sec=900.0.