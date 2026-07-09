# Task Instruction

## Task: Fill in training feedback HWPX template

### Goal
Replace all `{{...}}` placeholders in `training_feedback_template.hwpx` using values from `training_feedback.json`, apply the required transformations, and save the result to `/root/training_feedback_ready.hwpx`.

### Step-by-step plan

#### 1. Understand the HWPX format
A `.hwpx` file is a ZIP-based package (like DOCX/XLSX). It contains XML files inside. You need to:
- Unzip the template
- Find and edit the XML content files that contain the `{{...}}` placeholders
- Rezip into a valid `.hwpx` package

#### 2. Inspect the inputs
```bash
cd /root
cat training_feedback.json
```
Read the JSON carefully. Note every key and value.

Then explore the HWPX structure:
```bash
mkdir -p /tmp/hwpx_work
cp training_feedback_template.hwpx /tmp/hwpx_work/
cd /tmp/hwpx_work
unzip -o training_feedback_template.hwpx -d template_extracted
find template_extracted -type f
```

#### 3. Find all placeholders
Search every XML file for `{{` patterns:
```bash
grep -rn '{{' template_extracted/
```
Record every placeholder found and which file it's in. Map each placeholder to the corresponding JSON key.

#### 4. Apply replacements with transformations
Write a Python script that:

a. Loads `training_feedback.json`

b. For each XML file containing placeholders, reads the file content as UTF-8 text.

c. Replaces each `{{key}}` with the corresponding JSON value, applying these special rules:
   - **`참석자수` (attendee count)**: Extract digits only. E.g., if JSON has `"25명"` or `"25"`, write just `"25"` (digits only, no unit).
   - **`만족도` (satisfaction)**: Rewrite as `"X.X점 (5.0점 만점)"` format, where X.X is the numeric score from JSON. E.g., if JSON has `4.5` or `"4.5/5.0"` or `"4.5점"`, output `"4.5점 (5.0점 만점)"`.
   - **Overall opinion / 종합의견**: Find the placeholder for the overall comment. After inserting the JSON value, append ` 후속 심화반 검토 요망.` (with a space before it if needed). Make sure the appended text is part of the same text run or paragraph.
   - **All other placeholders**: Direct substitution with the JSON string value.

d. **Important**: After modifying any paragraph's text content, remove any layout-cache / char-position-cache elements from that paragraph. Specifically look for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hc:lineseg>`, or similar layout cache tags within or associated with modified paragraphs. Remove them so the document renders cleanly. Search the XML namespace prefixes used in the actual files to identify the correct element names.

e. Writes the modified XML back.

#### 5. Verify no placeholders remain
```bash
grep -rn '{{' template_extracted/
```
This must return nothing.

#### 6. Repackage as HWPX
The HWPX (like OPC/ZIP packages) may require `mimetype` as the first uncompressed entry. Check if a `mimetype` file exists:
```bash
ls template_extracted/mimetype 2>/dev/null
```

Repackage:
```python
import zipfile, os

output_path = '/root/training_feedback_ready.hwpx'
extracted_dir = '/tmp/hwpx_work/template_extracted'

with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    # If mimetype exists, add it first with no compression
    mimetype_path = os.path.join(extracted_dir, 'mimetype')
    if os.path.exists(mimetype_path):
        zf.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)
    
    for root, dirs, files in os.walk(extracted_dir):
        for f in sorted(files):
            full = os.path.join(root, f)
            arcname = os.path.relpath(full, extracted_dir)
            if arcname == 'mimetype':
                continue
            zf.write(full, arcname)
```

#### 7. Final validation
```bash
# Verify it's a valid ZIP
python3 -c "import zipfile; z=zipfile.ZipFile('/root/training_feedback_ready.hwpx'); z.testzip(); print('Valid ZIP'); z.close()"

# Verify no placeholders in the output
mkdir -p /tmp/hwpx_verify
unzip -o /root/training_feedback_ready.hwpx -d /tmp/hwpx_verify
grep -rn '{{' /tmp/hwpx_verify/

# Verify the transformations
grep -rn '점 (5.0점 만점)' /tmp/hwpx_verify/
grep -rn '후속 심화반 검토 요망' /tmp/hwpx_verify/
```

### Critical details to watch for
- The XML may split `{{placeholder}}` across multiple XML elements/tags. If `grep` shows `{{` but the placeholder text is split by XML tags (e.g., `<hp:t>{{참석</hp:t><hp:t>자수}}</hp:t>`), you must handle this by working at a higher level — perhaps collapsing adjacent text runs before replacing, or using regex that spans tags. **Check for this explicitly.**
- Namespace prefixes vary. Inspect the actual XML to find the correct tag names for text content and layout cache elements.
- The layout cache elements to remove: look for any `<*:linesegarray>` or `<*:lineSeg>` or similar within `<hp:p>` (paragraph) elements that you modify. Inspect the actual XML structure to identify them precisely.
- Preserve all Korean labels and static note lines — only replace `{{...}}` patterns.
- Ensure UTF-8 encoding is maintained throughout.

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