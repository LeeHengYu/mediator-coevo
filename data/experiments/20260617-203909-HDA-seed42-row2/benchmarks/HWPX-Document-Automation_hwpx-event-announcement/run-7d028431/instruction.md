# Task Instruction

Prepare the event announcement document by replacing all `{{...}}` placeholders with values from `event_data.json`, then save the result to `/root/event_announcement_ready.hwpx`.

## Step-by-step plan

### 1. Inspect the workspace
```bash
ls /root/
ls /root/HWPX-Document-Automation/hwpx-event-announcement/ 2>/dev/null || ls /root/hwpx-event-announcement/ 2>/dev/null || find /root -name 'event_data.json' -o -name 'event_announcement_template.hwpx' 2>/dev/null
```
Locate `event_announcement_template.hwpx` and `event_data.json`.

### 2. Read the JSON data
```bash
cat <path_to>/event_data.json
```
Note every key-value pair. These are the substitution values for `{{key}}` placeholders.

### 3. Explore the HWPX template structure
HWPX is a ZIP-based (OPC/OCF) package. Unzip it to a temporary directory:
```bash
mkdir -p /tmp/hwpx_work
cd /tmp/hwpx_work
unzip -o <path_to>/event_announcement_template.hwpx -d template_contents
find template_contents -type f
```
Pay special attention to:
- `mimetype` (must be stored uncompressed when repacking)
- `Contents/section*.xml` — these contain the document body text
- Any other XML files that might contain placeholders

### 4. Search for ALL placeholders across the entire package
```bash
grep -r '{{' template_contents/
```
This finds every `{{...}}` occurrence regardless of which XML file it lives in. Record which files need editing.

### 5. Inspect the XML files containing placeholders
Read each file that has `{{`. Check whether placeholders might be **split across XML tags** (e.g., `{{` in one `<hp:t>` element and `}}` in another, or inline XML tags like `<hp:markpenBegin/>` breaking up the placeholder text). This is critical — if the placeholder text is fragmented across tags, you must reassemble the combined text content of the parent run element, perform the substitution, then rewrite it as a single text node.

### 6. Write a Python script to perform the substitution
Create `/tmp/hwpx_work/process.py` that:

1. Loads `event_data.json`.
2. For each XML file containing `{{`:
   a. Parses it as XML (use `lxml.etree` if available, else `xml.etree.ElementTree`).
   b. Walks all text-bearing elements. For each parent element that contains child runs/text spans:
      - Concatenate all text content (`.text` and `.tail` of children) to detect `{{...}}` patterns.
      - If a placeholder is found spanning multiple child elements, merge the text into the first text node and clear the others.
   c. Performs regex substitution: for every `{{key}}`, replace with the corresponding value from the JSON. Use `re.sub(r'\{\{(.+?)\}\}', lambda m: data.get(m.group(1), m.group(0)), text)`.
   d. **Removes stale layout-cache elements**: For any paragraph (`<hp:p>`) whose text was modified, remove `<hp:lineSegArray>` (and its children) and `<hp:linesegarray>` elements. This prevents overlapping characters when the document is opened. Also remove any `<hp:LineSeg>` or similar layout cache sub-elements.
   e. Writes the modified XML back, preserving the XML declaration and encoding.
3. Verifies that NO `{{` remains in any XML file after processing.

### 7. Repackage the HWPX file
Repackage using Python's `zipfile` module to ensure correct OPC conventions:
```python
import zipfile, os

output_path = '/root/event_announcement_ready.hwpx'
base_dir = 'template_contents'

with zipfile.ZipFile(output_path, 'w') as zf:
    # Add mimetype FIRST, uncompressed (STORED)
    mimetype_path = os.path.join(base_dir, 'mimetype')
    if os.path.exists(mimetype_path):
        zf.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)
    
    # Add all other files with deflate compression
    for root, dirs, files in os.walk(base_dir):
        for f in sorted(files):
            full = os.path.join(root, f)
            arcname = os.path.relpath(full, base_dir)
            if arcname == 'mimetype':
                continue
            zf.write(full, arcname, compress_type=zipfile.ZIP_DEFLATED)
```

### 8. Validate the output
```bash
# Verify it's a valid ZIP
unzip -t /root/event_announcement_ready.hwpx

# Verify no placeholders remain
mkdir -p /tmp/hwpx_verify
unzip -o /root/event_announcement_ready.hwpx -d /tmp/hwpx_verify
grep -r '{{' /tmp/hwpx_verify/ && echo 'FAIL: placeholders remain' || echo 'PASS: no placeholders'

# Verify mimetype is first entry and uncompressed
python3 -c "
import zipfile
zf = zipfile.ZipFile('/root/event_announcement_ready.hwpx')
first = zf.infolist()[0]
print(f'First entry: {first.filename}, compress_type: {first.compress_type}')
assert first.filename == 'mimetype' and first.compress_type == 0, 'mimetype must be first and STORED'
print('mimetype check PASS')
"
```

### 9. Run the verifier if available
```bash
cd <task_directory>
python3 -m pytest test_output.py -v 2>&1 || true
```

## Key pitfalls to avoid
- **Split placeholders**: `{{event_name}}` might be split as `<t>{{event</t><t>_name}}</t>`. You MUST handle this by merging sibling text nodes before substitution.
- **Stale layout cache**: Any `<hp:lineSegArray>` in a modified paragraph MUST be removed.
- **mimetype entry**: Must be the first ZIP entry, stored uncompressed (ZIP_STORED).
- **Korean text preservation**: Do not alter any Korean labels or the static note line — only replace `{{...}}` patterns.
- **Namespace handling**: HWPX XML uses namespaces (e.g., `hp:`, `hc:`). Make sure your XML parsing preserves all namespace prefixes and declarations.

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