# Task Instruction

## Task: Fill in training feedback HWPX template

You need to fill in a HWPX template document with values from a JSON file and save the result.

### Step 0: Understand the HWPX format
A `.hwpx` file is a ZIP-based package (like .docx/.xlsx) containing XML files. The document content is typically in XML files inside the archive. You will need to:
- Unzip the template
- Find and edit the XML content files
- Re-zip into a valid .hwpx package

### Step 1: Inspect the input files
1. Read `/root/training_feedback.json` to understand all available values.
2. List the contents of `/root/training_feedback_template.hwpx` by treating it as a ZIP archive:
   ```python
   import zipfile
   with zipfile.ZipFile('/root/training_feedback_template.hwpx', 'r') as z:
       z.printdir()
   ```
3. Extract all files to a temporary directory `/root/hwpx_work/`.
4. Find all XML files in the extracted contents and search for `{{` placeholders across ALL of them. Print each file's content that contains placeholders so you can see the full XML structure.

### Step 2: Map JSON values to placeholders
For each `{{...}}` placeholder found, identify the corresponding JSON key. Pay special attention to:
- **`참석자수`** (number of attendees): Convert to digits only (e.g., if JSON has "25명" or similar, output just "25"; if it's already a number, use the number as a string of digits).
- **`만족도`** (satisfaction): Rewrite as `X.X점 (5.0점 만점)` format, where X.X is the numeric score from JSON.
- **Overall opinion sentence**: Find the placeholder for the overall comment/opinion and append ` 후속 심화반 검토 요망.` after the JSON-provided comment text (with a space before 후속).

### Step 3: Perform replacements in XML files
Write a Python script that:
1. Reads each XML file that contains placeholders.
2. Replaces every `{{...}}` placeholder with the correct formatted value.
3. **Critical**: After modifying any paragraph's text content, remove any layout-cache elements (commonly `<hp:linesegarray>`, `<lineseg>`, `<hp:lineSegArray>`, or similar cached layout elements) from that paragraph. These are pre-computed glyph positions that become stale after text changes and cause overlapping characters. Search for elements like `lineSegArray`, `lineSeg`, `linesegarray`, or similar layout cache nodes within modified paragraphs and remove them entirely.
4. Verify no `{{` or `}}` remain in any XML file after replacement.

### Step 4: Repackage as HWPX
Re-create the ZIP archive as `/root/training_feedback_ready.hwpx`:
```python
import zipfile, os

with zipfile.ZipFile('/root/training_feedback_ready.hwpx', 'w', zipfile.ZIP_DEFLATED) as zout:
    for root, dirs, files in os.walk('/root/hwpx_work/'):
        for f in files:
            full = os.path.join(root, f)
            arcname = os.path.relpath(full, '/root/hwpx_work/')
            zout.write(full, arcname)
```

### Step 5: Validate the output
1. Open `/root/training_feedback_ready.hwpx` as a ZIP and verify it's valid.
2. Read all XML content files from the output and confirm:
   - No `{{` or `}}` strings remain anywhere.
   - `참석자수` value is digits only (no Korean unit suffixes).
   - `만족도` appears in the format `X.X점 (5.0점 만점)`.
   - The overall opinion text ends with `후속 심화반 검토 요망.`
   - All Korean labels and static note lines are unchanged.
   - Layout cache elements have been removed from modified paragraphs.
3. Print a summary of all replacements made.

### Important constraints
- Do NOT change any Korean labels or static text that isn't a placeholder.
- Do NOT leave any `{{...}}` unreplaced.
- The output must be a valid ZIP/HWPX package.
- Remove stale layout-cache elements from edited paragraphs so the document opens cleanly.
- Inspect the actual XML namespace prefixes used in the template before writing removal code — don't assume namespace prefixes.

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