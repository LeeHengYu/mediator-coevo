# Task Instruction

Complete the following task step by step:

## Goal
Prepare the event announcement document by replacing all `{{...}}` placeholders in `event_announcement_template.hwpx` with values from `event_data.json`, then save the result to `/root/event_announcement_ready.hwpx`.

## Steps

### 1. Inspect the workspace
- List files in the current directory and `/root/` to locate `event_announcement_template.hwpx` and `event_data.json`.
- Read `event_data.json` to understand the key-value pairs available for substitution.

### 2. Understand the HWPX structure
- HWPX files are ZIP archives containing XML files. Use Python's `zipfile` module to list all entries in the template.
- Identify which XML files inside the archive contain `{{` placeholder text (typically files under `Contents/` such as `section0.xml`).
- Print the raw XML content of those files to see all placeholders and understand the document structure.

### 3. Write a Python script to perform the transformation
Create and run a Python script (`/root/process_hwpx.py`) that does the following:

```python
import json
import zipfile
import re
import os
import shutil
from lxml import etree

# Load JSON data
with open('event_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

template_path = 'event_announcement_template.hwpx'
output_path = '/root/event_announcement_ready.hwpx'

# Read all files from the template ZIP
with zipfile.ZipFile(template_path, 'r') as zin:
    file_contents = {}
    for name in zin.namelist():
        file_contents[name] = zin.read(name)

# Process each file in the archive
for name in list(file_contents.keys()):
    raw = file_contents[name]
    # Only process XML files that contain placeholders
    try:
        text = raw.decode('utf-8')
    except (UnicodeDecodeError, AttributeError):
        continue
    
    if '{{' not in text:
        continue
    
    # Replace all {{key}} placeholders with JSON values
    for key, value in data.items():
        placeholder = '{{' + key + '}}'
        text = text.replace(placeholder, str(value))
    
    # Also handle any placeholders that might be split across XML elements
    # by checking for remaining {{ patterns
    remaining = re.findall(r'\{\{.*?\}\}', text)
    if remaining:
        print(f"WARNING: Unreplaced placeholders in {name}: {remaining}")
    
    # Parse as XML to remove lineSegArray from modified paragraphs
    # Parse the XML
    root = etree.fromstring(text.encode('utf-8'))
    
    # Define namespace map from root
    nsmap = root.nsmap
    
    # Remove all hp:lineSegArray elements (layout cache) to prevent overlapping text
    # Find them using various possible namespace prefixes
    for elem in root.iter():
        local = etree.QName(elem.tag).localname if '}' in str(elem.tag) else elem.tag
        if local == 'lineSegArray':
            parent = elem.getparent()
            if parent is not None:
                parent.remove(elem)
    
    # Serialize back
    text = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=False)
    file_contents[name] = text

# Write the output HWPX file
with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
    for name, content in file_contents.items():
        if isinstance(content, str):
            content = content.encode('utf-8')
        zout.writestr(name, content)

print(f"Output written to {output_path}")
```

**Important considerations when writing the script:**
- Before running, first manually inspect the XML to see the exact placeholder names and compare them against the JSON keys. Adjust the script if keys don't match exactly (e.g., nested JSON, different naming conventions).
- If placeholders span across multiple XML elements (e.g., `<hp:t>{{</hp:t><hp:t>name}}</hp:t>`), you need to handle this by working at the text level before XML parsing, or by concatenating text runs.
- Preserve the original ZIP structure including `mimetype` file (which should be stored uncompressed if the original has it that way). Check the original ZIP's compression settings.

### 4. Handle edge cases
- If the JSON has nested objects, flatten them or map them appropriately to the placeholder names.
- Ensure Korean text and static note lines are untouched — only replace `{{...}}` patterns.
- After XML serialization, verify the XML declaration and encoding match the original.

### 5. Validate the output
- Open the output HWPX with `zipfile` and list its contents to confirm it's a valid ZIP.
- Read the XML files from the output and verify:
  - No `{{` or `}}` patterns remain anywhere in any file.
  - All Korean labels are preserved.
  - The `lineSegArray` elements have been removed from modified paragraphs.
- Print the full text content of the modified XML sections for visual confirmation.

### 6. Run the verifier
- If there is a test file (e.g., `test_output.py` or similar), run it with `pytest` to confirm the solution passes.
- Check: `python -m pytest /root/*.py -v` or look for test files in the task directory.

## Critical Notes
- The `lineSegArray` removal is essential — without it, HWPX viewers show overlapping characters on modified paragraphs.
- The output must be at exactly `/root/event_announcement_ready.hwpx`.
- Do NOT modify Korean labels or static note lines — only replace `{{...}}` placeholders.
- Ensure mimetype entry in the ZIP is handled correctly (often first entry, stored without compression).

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