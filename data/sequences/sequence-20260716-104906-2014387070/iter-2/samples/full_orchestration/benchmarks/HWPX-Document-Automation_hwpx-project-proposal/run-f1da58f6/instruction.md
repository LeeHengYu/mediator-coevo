# Task Instruction

Complete the HWPX document template with JSON values. Follow these steps precisely:

## Step 1: Inspect the workspace
```bash
ls /root/
cat /root/project_proposal.json
```

## Step 2: Examine the HWPX structure
HWPX files are ZIP archives. Extract and inspect:
```bash
mkdir -p /tmp/hwpx_work
cd /tmp/hwpx_work
unzip /root/project_proposal_template.hwpx -d template_contents/
find template_contents/ -type f
```

## Step 3: Find all files containing placeholders
```bash
grep -rl '{{' template_contents/
```
Then inspect each file that contains `{{...}}` placeholders to understand the XML structure.

## Step 4: Write a Python script to perform the transformation
Create `/tmp/hwpx_work/transform.py` that does the following:

### 4a: Load the JSON data
Read `/root/project_proposal.json` and build a mapping of placeholder keys to values.

### 4b: Budget normalization
For any budget/cost value, remove commas from the numeric part while keeping the leading currency symbol (e.g., `₩1,000,000` → `₩1000000`). Identify which JSON key(s) correspond to budget fields and normalize them.

### 4c: Process each XML file containing placeholders
For each XML file with `{{...}}`:
1. Parse it (use raw text manipulation or lxml, whichever is more reliable for preserving the XML structure).
2. Replace every `{{placeholder_name}}` with the corresponding JSON value.
3. For phase lines containing `단계1`, `단계2`, `단계3`: parse the date range already present in that line, calculate the month span between start and end dates, and append ` (N개월)` after the phase content. The expected results based on the benchmark are: 단계1 → (3개월), 단계2 → (3개월), 단계3 → (1개월). Calculate these from the actual dates to verify.
4. Remove stale layout-cache elements from any paragraph whose text was modified. In HWPX XML, these are typically `<hp:linesegarray>` elements (or elements in the `lineseg` namespace). Remove the entire element and its children from modified paragraphs.
5. Verify no `{{...}}` patterns remain anywhere in the file.

### 4d: Repackage as HWPX
Create the output ZIP file at `/root/project_proposal_ready.hwpx`:
- Use `zipfile` module in Python
- Preserve the original directory structure
- Use `ZIP_DEFLATED` compression
- Ensure `mimetype` file (if present) is stored first and uncompressed (like ODF convention), or match the original archive's compression settings

## Step 5: Run the transformation
```bash
cd /tmp/hwpx_work
python3 transform.py
```

## Step 6: Validate the output
```bash
# Verify it's a valid ZIP
unzip -t /root/project_proposal_ready.hwpx

# Extract and check for remaining placeholders
mkdir -p /tmp/hwpx_work/output_check
unzip /root/project_proposal_ready.hwpx -d /tmp/hwpx_work/output_check/
grep -r '{{' /tmp/hwpx_work/output_check/ && echo 'FAIL: placeholders remain' || echo 'PASS: no placeholders'

# Check that phase lines have month spans
grep -r '개월' /tmp/hwpx_work/output_check/

# Check budget value has no commas (but has currency symbol)
grep -r '₩' /tmp/hwpx_work/output_check/ || grep -r 'won\|원\|budget' /tmp/hwpx_work/output_check/

# Check that linesegarray elements are removed from modified paragraphs
# (inspect a sample modified paragraph)
```

## Important details:
- **Month span calculation**: Parse the date strings (likely in YYYY.MM.DD or YYYY-MM-DD format) from each phase line. Calculate months as: (end_year - start_year) * 12 + (end_month - start_month). If the end day is after the start day, you may need to add 1 month depending on convention. Match the expected outputs: 단계1→3개월, 단계2→3개월, 단계3→1개월.
- **Layout cache cleanup**: In HWPX XML, look for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, or any element whose tag contains `lineseg` (case-insensitive). Remove these from any `<hp:p>` (paragraph) element where text content was modified.
- **Placeholder replacement**: Placeholders may span across XML element boundaries (e.g., `<hp:t>{{</hp:t><hp:t>name}}</hp:t>`). If this occurs, you need to handle cross-element placeholders. First check if they're contained within single elements; if not, concatenate adjacent text runs, replace, and put the result in the first run while clearing subsequent ones.
- **Korean text preservation**: Do not modify any Korean label text or the static note line. Only replace `{{...}}` patterns and append month spans.
- **Encoding**: Ensure UTF-8 encoding is preserved in all XML files.

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
Task metadata: author_email=catpaw@example.com, author_name=CatPaw Task Engineer, category=document-editing, difficulty=medium, tags=[hwpx, xml-editing, document-processing, latent-method-reuse].
Verifier config: timeout_sec=600.0.