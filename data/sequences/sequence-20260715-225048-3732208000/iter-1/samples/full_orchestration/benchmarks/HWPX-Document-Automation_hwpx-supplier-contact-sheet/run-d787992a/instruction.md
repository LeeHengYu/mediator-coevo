# Task Instruction

Complete the following task step by step.

## Objective
Update the HWPX supplier contact sheet `supplier_contact_template.hwpx` using the values in `supplier_contact.json`, then save the finished file to `/root/supplier_contact_ready.hwpx`.

## Steps

### 1. Explore the workspace
```bash
find /root -maxdepth 2 -type f | head -60
ls -la /root/
```
Locate `supplier_contact_template.hwpx` and `supplier_contact.json`.

### 2. Read the JSON data
```bash
cat /root/supplier_contact.json
```
Note every key-value pair. These are the replacements for `{{key}}` placeholders.

### 3. Understand the HWPX package structure
A `.hwpx` file is a ZIP archive. Unzip it to inspect its contents:
```bash
mkdir -p /tmp/hwpx_work
cp /root/supplier_contact_template.hwpx /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
unzip -o template.hwpx -d template_extracted
find template_extracted -type f
```

### 4. Identify which XML files contain placeholders
Search for `{{` in all extracted files:
```bash
grep -rl '{{' template_extracted/
```
Then for each file found, display its contents:
```bash
for f in $(grep -rl '{{' template_extracted/); do echo "=== $f ==="; cat "$f"; echo; done
```

### 5. Understand the placeholder-to-JSON mapping
Each `{{placeholder_name}}` in the XML must be replaced with the corresponding value from `supplier_contact.json`. Map them carefully. The JSON keys should match the placeholder names (possibly with dots or underscores).

### 6. Perform replacements with a Python script
Write and run a Python script that:
- Loads the JSON file.
- For each XML file in the extracted HWPX directory that contains `{{...}}`:
  - Reads the file content as UTF-8 text.
  - For each key in the JSON, replaces `{{key}}` with the corresponding value. Handle nested JSON by flattening if needed (e.g., `contact.name` -> value).
  - **Critical**: After replacing text in any paragraph (`<hp:p>` element or similar), remove any layout-cache / char-position-cache elements. Specifically, look for elements like `<hp:linesegarray>`, `<hp:lineseg>`, `<hc:lineseg>`, `<hp:lineSegArray>`, or similar layout cache elements within the same paragraph and remove them entirely. These elements cache glyph positions and become stale after text changes, causing overlapping characters.
  - Writes the modified content back.
- After all replacements, verifies NO `{{...}}` patterns remain in any file.
- Preserves Korean field labels and the static note line (they should be untouched since they don't contain `{{}}` patterns, but verify).

Here is a template for the script:
```python
import json
import os
import re
import zipfile
import shutil

# Load JSON
with open('/root/supplier_contact.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Flatten nested JSON if needed
def flatten_json(obj, prefix=''):
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, (dict, list)):
                items.update(flatten_json(v, new_key))
            else:
                items[new_key] = str(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                items.update(flatten_json(v, new_key))
            else:
                items[new_key] = str(v)
    return items

# Try both flat and non-flat keys
flat_data = flatten_json(data)
# Also keep top-level simple keys
simple_data = {k: str(v) for k, v in data.items() if not isinstance(v, (dict, list))}
all_replacements = {**flat_data, **simple_data}

print("Replacement keys:", list(all_replacements.keys()))

extracted_dir = '/tmp/hwpx_work/template_extracted'

# Find all files with placeholders
modified_files = []
for root, dirs, files in os.walk(extracted_dir):
    for fname in files:
        fpath = os.path.join(root, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, IsADirectoryError):
            continue
        
        if '{{' not in content:
            continue
        
        print(f"\nProcessing: {fpath}")
        
        # Find all placeholders in this file
        placeholders = re.findall(r'\{\{([^}]+)\}\}', content)
        print(f"  Placeholders found: {placeholders}")
        
        original_content = content
        
        # Replace placeholders
        for ph in placeholders:
            pattern = '{{' + ph + '}}'
            # Try exact match first, then trimmed
            ph_stripped = ph.strip()
            if ph_stripped in all_replacements:
                val = all_replacements[ph_stripped]
                content = content.replace(pattern, val)
                print(f"  Replaced {pattern} -> {val}")
            elif ph in all_replacements:
                val = all_replacements[ph]
                content = content.replace(pattern, val)
                print(f"  Replaced {pattern} -> {val}")
            else:
                print(f"  WARNING: No replacement found for placeholder '{ph}'")
                # Try partial matching
                for key in all_replacements:
                    if key.endswith(ph_stripped) or ph_stripped.endswith(key):
                        val = all_replacements[key]
                        content = content.replace(pattern, val)
                        print(f"  Replaced {pattern} -> {val} (partial match on key '{key}')")
                        break
        
        # Remove layout cache elements from paragraphs that were modified
        # These are elements like <hp:linesegarray>...</hp:linesegarray> or <linesegarray>...</linesegarray>
        # that cache character positions and become stale after text edits
        if content != original_content:
            # Remove linesegarray elements (various namespace prefixes)
            content = re.sub(r'<[a-zA-Z]*:?linesegarray[^>]*>.*?</[a-zA-Z]*:?linesegarray>', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<linesegarray[^>]*>.*?</linesegarray>', '', content, flags=re.DOTALL | re.IGNORECASE)
            # Also try LineSeg variants
            content = re.sub(r'<[a-zA-Z]*:?lineSegArray[^>]*>.*?</[a-zA-Z]*:?lineSegArray>', '', content, flags=re.DOTALL)
            modified_files.append(fpath)
        
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)

# Verify no placeholders remain
print("\n=== Verification ===")
remaining = []
for root, dirs, files in os.walk(extracted_dir):
    for fname in files:
        fpath = os.path.join(root, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, IsADirectoryError):
            continue
        found = re.findall(r'\{\{[^}]+\}\}', content)
        if found:
            remaining.append((fpath, found))
            print(f"  REMAINING in {fpath}: {found}")

if remaining:
    print("ERROR: Some placeholders still remain!")
else:
    print("OK: No placeholders remain.")

print(f"\nModified files: {modified_files}")
```

### 7. Handle edge case: placeholders split across XML tags
After running the script, if any `{{...}}` placeholders remain, it may be because the placeholder text is split across multiple XML elements (e.g., `<run>{{</run><run>name</run><run>}}</run>`). If this happens:
- Parse the XML properly with `lxml` or `xml.etree.ElementTree`.
- Concatenate text within paragraph elements, find placeholders spanning multiple runs, and replace them while preserving the XML structure.
- Re-check and remove layout caches.

### 8. Repackage the HWPX file
Repackage the modified extracted directory back into a ZIP file with `.hwpx` extension:
```python
import zipfile
import os

output_path = '/root/supplier_contact_ready.hwpx'
extracted_dir = '/tmp/hwpx_work/template_extracted'

# Use the original zip to preserve compression methods and entry order
with zipfile.ZipFile('/tmp/hwpx_work/template.hwpx', 'r') as orig_zip:
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as out_zip:
        for item in orig_zip.infolist():
            file_path = os.path.join(extracted_dir, item.filename)
            if item.is_dir():
                out_zip.mkdir(item.filename)
                continue
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    out_zip.writestr(item, f.read())
            else:
                # File might not have been extracted (unlikely)
                out_zip.writestr(item, orig_zip.read(item.filename))

print(f"Output written to {output_path}")
```

### 9. Final verification
```bash
# Verify the output exists and is a valid zip
file /root/supplier_contact_ready.hwpx
unzip -t /root/supplier_contact_ready.hwpx

# Verify no placeholders remain in the output
mkdir -p /tmp/hwpx_verify
unzip -o /root/supplier_contact_ready.hwpx -d /tmp/hwpx_verify
grep -r '{{' /tmp/hwpx_verify/ || echo "No placeholders found - GOOD"

# Show the content of modified XML files to confirm Korean labels are preserved and values are filled in
for f in $(find /tmp/hwpx_verify -name '*.xml' -o -name '*.xhtml'); do echo "=== $f ==="; cat "$f"; echo; done
```

### Important notes
- **Korean labels**: Do NOT modify any Korean text that is not inside `{{...}}` placeholders. The Korean labels (like 업체명, 담당자, etc.) must remain.
- **Static note line**: Any line that doesn't contain placeholders should remain unchanged.
- **Layout caches**: After modifying paragraph text, remove `linesegarray` (and similar layout-cache elements) from those paragraphs. This is critical for the document to render correctly.
- **Valid HWPX**: The output must be a proper ZIP file with the same structure as the input.
- If the initial simple replacement approach leaves behind any `{{...}}` patterns, investigate whether placeholders are split across XML elements and handle that case with proper XML parsing.

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