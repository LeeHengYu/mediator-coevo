# Task Instruction

Complete the following task step by step.

## Goal
Prepare the event announcement document `event_announcement_template.hwpx` using the values in `event_data.json`, then save the result to `/root/event_announcement_ready.hwpx`.

## Steps

### 1. Understand the HWPX format
A `.hwpx` file is a ZIP archive containing XML files (similar to OOXML). The main document content is typically in an XML file inside the archive (often `Contents/section0.xml` or similar). Explore the archive structure first.

### 2. Inspect the template and data
```bash
cd /root
cat event_data.json
```
Then explore the HWPX archive:
```bash
python3 -c "
import zipfile, os
with zipfile.ZipFile('event_announcement_template.hwpx', 'r') as z:
    for info in z.infolist():
        print(f'{info.file_size:>8}  {info.filename}')
"
```

### 3. Find all files containing `{{` placeholders
Extract and search every text/XML file in the archive for `{{` to identify which files need modification:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('event_announcement_template.hwpx', 'r') as z:
    for name in z.namelist():
        try:
            data = z.read(name).decode('utf-8')
            if '{{' in data:
                print(f'=== {name} ===')
                # Print lines containing {{
                for i, line in enumerate(data.split('\n')):
                    if '{{' in line:
                        print(f'  Line {i}: ...{line.strip()[:200]}...')
        except:
            pass
"
```

### 4. Read event_data.json and perform replacements
Write a Python script that:
1. Reads `event_data.json` to get the key-value mapping.
2. Opens the HWPX ZIP archive.
3. For each file in the archive, if it's a text/XML file containing `{{...}}` placeholders, replaces every `{{key}}` with the corresponding value from the JSON.
4. **Critical**: After replacing text in paragraph elements, remove any layout-cache elements (commonly `<hp:linesegarray>`, `<hp:lineSegArray>`, or similar elements that cache glyph/character positions) from the same paragraph. These stale caches cause overlapping characters when the document is opened. Look for elements like `<hp:linesegarray>` or any element whose tag contains `lineseg`, `lineSegArray`, `LineSeg`, or similar layout cache tags. Remove them entirely.
5. Writes all files (modified and unmodified) to the output `/root/event_announcement_ready.hwpx`, preserving the original ZIP structure and compression.

### 5. Implementation script
```python
import zipfile
import json
import re
import os
from io import BytesIO

# Load event data
with open('event_data.json', 'r', encoding='utf-8') as f:
    event_data = json.load(f)

# Build replacement map: {{key}} -> value
# Handle both flat and nested JSON structures
def flatten_json(obj, prefix=''):
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f'{prefix}{k}' if not prefix else f'{prefix}.{k}'
            if isinstance(v, (dict, list)):
                items.update(flatten_json(v, new_key))
            else:
                items[new_key] = str(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items.update(flatten_json(v, f'{prefix}[{i}]'))
    return items

# First try flat keys, then nested
replacements = {}
if isinstance(event_data, dict):
    # Add top-level keys directly
    for k, v in event_data.items():
        if not isinstance(v, (dict, list)):
            replacements[k] = str(v)
    # Also add flattened nested keys
    flat = flatten_json(event_data)
    replacements.update(flat)

print('Replacement keys:', list(replacements.keys()))

# Process HWPX
input_path = 'event_announcement_template.hwpx'
output_path = '/root/event_announcement_ready.hwpx'

with zipfile.ZipFile(input_path, 'r') as zin:
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            
            # Try to process as text
            try:
                text = data.decode('utf-8')
                if '{{' in text:
                    print(f'Processing: {item.filename}')
                    # Replace all {{key}} patterns
                    for key, value in replacements.items():
                        text = text.replace('{{' + key + '}}', value)
                    
                    # Check for any remaining placeholders
                    remaining = re.findall(r'\{\{[^}]+\}\}', text)
                    if remaining:
                        print(f'  WARNING: Remaining placeholders: {remaining}')
                    
                    # Remove layout cache elements (linesegarray and similar)
                    # These are XML elements that cache character positions
                    # Use regex to remove them since they may span multiple lines
                    # Common patterns: <hp:linesegarray>...</hp:linesegarray>
                    # Also: <linesegarray>...</linesegarray>, <hp:LineSeg>...</hp:LineSeg>
                    for tag_pattern in [
                        r'<[a-zA-Z]*:?linesegarray[^>]*>.*?</[a-zA-Z]*:?linesegarray>',
                        r'<[a-zA-Z]*:?lineSegArray[^>]*>.*?</[a-zA-Z]*:?lineSegArray>',
                        r'<[a-zA-Z]*:?LineSeg[^>]*>.*?</[a-zA-Z]*:?LineSeg>',
                        r'<[a-zA-Z]*:?lineseg[^>]*>.*?</[a-zA-Z]*:?lineseg>',
                    ]:
                        text = re.sub(tag_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
                    
                    data = text.encode('utf-8')
            except (UnicodeDecodeError, Exception):
                pass
            
            zout.writestr(item, data)

print('Output written to', output_path)
```

### 6. Validate the output
After creating the output file, verify:
```bash
python3 -c "
import zipfile, re
with zipfile.ZipFile('/root/event_announcement_ready.hwpx', 'r') as z:
    print('Valid ZIP: yes')
    print('Files:', len(z.namelist()))
    for name in z.namelist():
        try:
            data = z.read(name).decode('utf-8')
            remaining = re.findall(r'\{\{[^}]+\}\}', data)
            if remaining:
                print(f'  REMAINING PLACEHOLDERS in {name}: {remaining}')
        except:
            pass
    print('Validation complete')
"
```

### Important Notes
- **Do NOT change Korean labels or static note lines** — only replace `{{...}}` placeholders.
- **Remove ALL layout-cache/linesegarray elements** from any paragraph whose text was modified. This is critical for the document to render correctly.
- If the JSON has nested structure, inspect it carefully and match the placeholder keys exactly as they appear in the template XML.
- If placeholders use dot notation (e.g., `{{event.name}}`), make sure to handle nested JSON keys accordingly.
- Ensure the output is a valid ZIP/HWPX by using Python's zipfile module properly.
- After initial exploration, adapt the script if the XML structure or placeholder format differs from expectations.

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