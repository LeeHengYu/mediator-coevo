# Task Instruction

Complete the following task to prepare an event announcement HWPX document.

## Goal
Fill in all `{{...}}` placeholders in `event_announcement_template.hwpx` using values from `event_data.json`, and save the result to `/root/event_announcement_ready.hwpx`.

## Steps

### 1. Inspect the workspace
```bash
ls -la /root/
cat /root/event_data.json
```
Identify the template file location and understand the JSON data structure (keys and values).

### 2. Explore the HWPX package structure
HWPX files are ZIP archives containing XML files. Extract and inspect:
```bash
cd /root
mkdir -p hwpx_temp
cp event_announcement_template.hwpx hwpx_temp/template.zip
cd hwpx_temp
unzip template.zip -d template_contents
find template_contents -type f
```
List all files in the archive to understand the structure.

### 3. Identify files containing placeholders
Search for `{{` in all extracted files:
```bash
grep -rl '{{' template_contents/
```
Then inspect each matching file to see the full context of placeholders:
```bash
grep -n '{{' template_contents/<each_matching_file>
```
Also cat the full content of the main content XML files (likely under `Contents/` directory, e.g., `section0.xml` or similar).

### 4. Write a Python script to perform the replacement
Create a Python script `/root/fill_template.py` that:

a. Reads `event_data.json` to get the replacement values.

b. Opens `event_announcement_template.hwpx` as a ZIP file.

c. Iterates through every file in the ZIP archive.

d. For each XML file, reads its content as UTF-8 text and performs string replacement of every `{{key}}` with the corresponding JSON value. For non-XML (binary) files, copies them as-is.

e. **Critical: For any XML file that contained placeholders and was modified, parse it with `lxml.etree` (or `xml.etree.ElementTree`) and remove all layout-cache elements.** In HWPX format, layout-cache elements are typically `<hp:linesegarray>` or elements within a `<hp:lineseg>` namespace, or elements like `<paramlist>` with layout cache data. Specifically:
   - Parse the modified XML with an XML parser that preserves namespaces.
   - Find all paragraphs (likely `<hp:p>` elements) that were modified (i.e., contained a placeholder).
   - Within those paragraphs, remove child elements related to layout caching. These are typically `<hp:linesegarray>` elements (or similar names). Look for elements whose local name is `linesegarray` or `lineSegArray` or similar.
   - Actually, to be safe: in each modified paragraph, remove ALL `<hp:linesegarray>` (or any element whose local name contains 'lineseg' or 'LineSeg') children.
   - Re-serialize the XML back to a UTF-8 string, preserving the XML declaration if one was present.

f. Writes all files (modified and unmodified) into a new ZIP file at `/root/event_announcement_ready.hwpx`, preserving the original directory structure and compression settings.

### 5. Detailed implementation guidance for the Python script

```python
import json
import zipfile
import os
import re
import copy
from lxml import etree

# Read JSON data
with open('/root/event_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

template_path = '/root/event_announcement_template.hwpx'
output_path = '/root/event_announcement_ready.hwpx'

with zipfile.ZipFile(template_path, 'r') as zin:
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            raw = zin.read(item.filename)
            
            # Try to process as text (XML) file
            try:
                content = raw.decode('utf-8')
            except (UnicodeDecodeError, AttributeError):
                zout.writestr(item, raw)
                continue
            
            # Check if this file has any placeholders
            if '{{' in content:
                # Replace all placeholders
                for key, value in data.items():
                    content = content.replace('{{' + key + '}}', str(value))
                
                # Also replace any remaining {{...}} patterns using nested keys or alternate formats
                # Verify no {{ remains
                
                # Now remove layout cache from modified paragraphs
                # Parse as XML
                try:
                    tree = etree.fromstring(content.encode('utf-8'))
                    nsmap = tree.nsmap
                    
                    # Find all elements whose local name suggests layout caching
                    # Common in HWPX: linesegarray, lineSegArray, LineSeg, etc.
                    # Remove them from paragraphs
                    for elem in tree.iter():
                        local = etree.QName(elem.tag).localname if isinstance(elem.tag, str) else ''
                        if local.lower() in ('linesegarray',):
                            parent = elem.getparent()
                            if parent is not None:
                                parent.remove(elem)
                    
                    # Re-serialize
                    xml_decl = content.startswith('<?xml')
                    content = etree.tostring(tree, xml_declaration=xml_decl, encoding='unicode')
                    # If original had specific encoding declaration, adjust
                    if xml_decl and 'encoding=' in content[:100]:
                        pass  # etree handles it
                except etree.XMLSyntaxError:
                    pass  # If not valid XML, just use string-replaced content
                
                zout.writestr(item, content.encode('utf-8'))
            else:
                zout.writestr(item, raw)
```

**Important adjustments to make during implementation:**
- Before writing the script, first inspect the actual XML files from step 3 to identify the exact element names used for layout caching. Adjust the element removal logic accordingly.
- Check if placeholders might be split across multiple XML elements (e.g., `<run>{{</run><run>event_name</run><run>}}</run>`). If so, you'll need to handle text concatenation within paragraphs before replacement.
- Preserve the original `ZipInfo` compression type if possible.

### 6. Run the script
```bash
python3 /root/fill_template.py
```

### 7. Validate the output
```bash
# Check it's a valid ZIP
unzip -t /root/event_announcement_ready.hwpx

# Check no placeholders remain
mkdir -p /root/hwpx_verify
cd /root/hwpx_verify
unzip /root/event_announcement_ready.hwpx -d verify_contents
grep -r '{{' verify_contents/ || echo 'No placeholders found - GOOD'

# Verify Korean labels are preserved (spot check a few known Korean strings from the template)
grep -r '행사' verify_contents/ || true

# Verify JSON values appear in output
# (Check for a few specific values from event_data.json)
```

If any `{{...}}` placeholders remain, investigate whether keys in the JSON don't match placeholder names, or whether placeholders are split across XML elements, and fix accordingly.

### 8. Final check
Confirm `/root/event_announcement_ready.hwpx` exists and is non-empty:
```bash
ls -la /root/event_announcement_ready.hwpx
```

## Critical Reminders
- The HWPX file must remain a valid ZIP package.
- ALL `{{...}}` placeholders must be replaced - zero may remain.
- Korean labels and the static note line must be unchanged.
- Layout-cache elements (`linesegarray` or similar) in modified paragraphs MUST be removed so the document opens cleanly.
- Inspect actual file contents before writing code - do not assume structure.
- If `lxml` is not available, use `xml.etree.ElementTree` instead (it lacks `getparent()`, so use a parent map pattern).

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