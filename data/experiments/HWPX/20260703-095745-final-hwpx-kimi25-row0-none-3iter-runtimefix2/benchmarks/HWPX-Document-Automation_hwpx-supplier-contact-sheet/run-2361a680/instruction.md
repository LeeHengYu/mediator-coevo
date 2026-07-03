# Task Instruction

Complete the following task to update a HWPX supplier contact sheet template with real data.

## Background
HWPX files are ZIP-based Korean word-processor document packages (Hancom Office). They contain XML files inside (similar to DOCX). The main content is typically in `Contents/section0.xml` (or similar paths). You need to:
1. Unzip the template
2. Find and replace all `{{...}}` placeholders with values from the JSON file
3. Clean up layout-cache elements for modified paragraphs
4. Repackage as a valid HWPX (ZIP) file

## Steps

### Step 1: Inspect the workspace
```bash
ls -la /root/
find /root/ -name 'supplier_contact_template.hwpx' -o -name 'supplier_contact.json' 2>/dev/null
```
Locate both `supplier_contact_template.hwpx` and `supplier_contact.json`.

### Step 2: Read the JSON data
```bash
cat supplier_contact.json
```
Note all key-value pairs. These are the replacement values for `{{key}}` placeholders.

### Step 3: Explore the HWPX structure
```bash
mkdir -p /tmp/hwpx_work
cp supplier_contact_template.hwpx /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
unzip -o template.hwpx -d template_extracted
find template_extracted -type f
```
List all files in the archive to understand the package structure.

### Step 4: Find all placeholders
```bash
grep -r '{{' template_extracted/ --include='*.xml' -l
grep -rn '{{[^}]*}}' template_extracted/ --include='*.xml'
```
Identify every file and location containing `{{...}}` placeholders. Also check non-XML files:
```bash
grep -r '{{' template_extracted/ -l
```

### Step 5: Examine the XML content files in detail
For each file containing placeholders, read its full contents:
```bash
cat template_extracted/Contents/section0.xml
```
(Adjust path as needed based on Step 4 results.)

Pay special attention to:
- How placeholders appear in the XML (they may be split across multiple XML elements/tags)
- Layout-cache elements (look for tags like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:LineSeg>`, `<hp:charShapeArray>`, or similar caching/layout elements within paragraph nodes)
- Korean field labels that must be preserved
- Static note lines that must remain unchanged

### Step 6: Write a Python script to perform the replacement
Write a Python script that:
1. Reads the JSON file to get replacement values
2. For each XML file containing placeholders:
   a. Reads the XML content
   b. Replaces every `{{key}}` with the corresponding JSON value
   c. **CRITICAL**: Handles cases where placeholders might be split across XML inline elements. If `{{placeholder}}` is split like `<run>{{place</run><run>holder}}</run>`, you need to reconstruct and replace across element boundaries. First check if placeholders appear intact or split.
   d. **CRITICAL**: For any paragraph (`<hp:p>` or similar paragraph element) whose text content was modified, remove layout-cache child elements. These are typically `<hp:linesegarray>` or `<hp:lineSegArray>` or similar elements that cache glyph positions. Removing them forces the word processor to recalculate layout on open, preventing overlapping characters.
   e. Writes the modified XML back
3. Verifies no `{{...}}` patterns remain in any file in the package

Here is a template for the script:
```python
import json
import os
import re
import shutil
from lxml import etree

# Read JSON
with open('/root/supplier_contact.json', 'r', encoding='utf-8') as f:  # adjust path if needed
    data = json.load(f)

extracted_dir = '/tmp/hwpx_work/template_extracted'

# Find all XML files with placeholders
for root, dirs, files in os.walk(extracted_dir):
    for fname in files:
        fpath = os.path.join(root, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        if '{{' not in content:
            continue
        
        print(f'Processing: {fpath}')
        print(f'Placeholders found: {re.findall(r"\{\{[^}]*\}\}", content)}')
        
        # Try XML-aware approach first
        # Parse, iterate text nodes, replace, remove layout caches
        # ... (implement based on actual XML structure observed)
        
        # Simple text replacement as baseline
        modified = content
        for key, value in data.items():
            placeholder = '{{' + key + '}}'
            if placeholder in modified:
                modified = modified.replace(placeholder, str(value))
                print(f'  Replaced {placeholder} -> {value}')
        
        # Check for remaining placeholders
        remaining = re.findall(r'\{\{[^}]*\}\}', modified)
        if remaining:
            print(f'  WARNING: Remaining placeholders: {remaining}')
        
        # Write back
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(modified)
```

**IMPORTANT**: After the simple replacement, you MUST handle layout cache removal. Parse the modified XML files with lxml (or xml.etree.ElementTree), find all paragraph elements that were modified, and remove their layout-cache children. The exact element names depend on the HWPX namespace and schema — inspect the XML to determine them. Common patterns:
- `<hp:linesegarray>` or `<hp:lineSegArray>` inside `<hp:p>` elements
- Any element that appears to store character position/width caching data

Remove these elements from paragraphs whose text was changed.

### Step 7: Remove layout cache elements
After replacement, parse each modified XML file and remove layout-cache elements from modified paragraphs. If unsure which paragraphs were modified, it is safe to remove layout-cache elements from ALL paragraphs — the application will regenerate them.

```python
# After text replacement, re-parse and clean layout caches
tree = etree.parse(fpath)
root_el = tree.getroot()
nsmap = root_el.nsmap

# Find all paragraph elements and remove lineseg arrays
# Adjust namespace prefix and element names based on actual XML inspection
for p in root_el.iter():
    if 'linesegarray' in p.tag.lower() or 'lineSegArray' in p.tag:
        parent = p.getparent()
        if parent is not None:
            parent.remove(p)
            print(f'  Removed layout cache: {p.tag}')

tree.write(fpath, xml_declaration=True, encoding='utf-8')
```

### Step 8: Verify no placeholders remain
```bash
grep -r '{{' /tmp/hwpx_work/template_extracted/
```
This must return no results. If any `{{...}}` remain, investigate and fix.

### Step 9: Repackage as HWPX
HWPX files are ZIP archives. Repackage carefully, preserving the directory structure:
```bash
cd /tmp/hwpx_work/template_extracted
zip -r /root/supplier_contact_ready.hwpx . -x '*.DS_Store'
```
Note: Use `zip` from within the extracted directory so paths are relative (no leading directory name).

### Step 10: Validate the output
```bash
# Verify it's a valid ZIP
file /root/supplier_contact_ready.hwpx
unzip -l /root/supplier_contact_ready.hwpx

# Verify no placeholders in the final package
mkdir -p /tmp/hwpx_verify
cd /tmp/hwpx_verify
unzip -o /root/supplier_contact_ready.hwpx
grep -r '{{' . || echo 'No placeholders found - PASS'

# Verify the file exists at the correct path
ls -la /root/supplier_contact_ready.hwpx
```

## Critical Requirements
- Every `{{...}}` placeholder must be replaced — zero may remain
- Korean field labels in the document must be preserved (do not translate or remove them)
- Static note lines must remain unchanged
- Layout-cache elements in modified paragraphs must be removed
- Output must be at `/root/supplier_contact_ready.hwpx`
- Output must be a valid ZIP (HWPX) package with correct internal structure

## Debugging Notes
- If placeholders are split across XML elements, you'll need to concatenate text across sibling elements, find the placeholder, replace it, and redistribute text back. This is the hardest part — inspect the XML carefully first.
- The JSON keys should match the placeholder names exactly (without the `{{` and `}}` delimiters)
- If you encounter encoding issues, ensure UTF-8 is used throughout
- When repackaging, make sure the ZIP mimetype entry (if present) is stored uncompressed (first entry, no compression) — check if the original has this pattern

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