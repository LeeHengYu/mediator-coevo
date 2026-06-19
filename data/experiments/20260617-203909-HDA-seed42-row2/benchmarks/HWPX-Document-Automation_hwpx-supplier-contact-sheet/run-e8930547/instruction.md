# Task Instruction

Complete the following task to update a HWPX supplier contact sheet template with actual data.

## Background
HWPX files are ZIP-based document packages (similar to DOCX) used by the Korean word processor Hancom/Hangul. Inside the ZIP, the main document content is typically in XML files (often `Contents/section0.xml` or similar). Placeholders like `{{COMPANY_NAME}}` appear in the XML text nodes and must be replaced with values from a JSON file.

## Steps

### 1. Inspect the workspace
```bash
ls -la /root/
cat /root/supplier_contact.json
```
Identify the template file `supplier_contact_template.hwpx` and read the JSON data file.

### 2. Examine the HWPX package structure
```bash
cd /root
mkdir -p hwpx_work
cp supplier_contact_template.hwpx hwpx_work/template.zip
cd hwpx_work
unzip -l template.zip
```
List all files inside the HWPX archive to understand the structure.

### 3. Extract the archive
```bash
unzip -o template.zip -d template_extracted
```

### 4. Find all placeholder locations
```bash
grep -r '{{' template_extracted/
```
This will show every file and line containing `{{...}}` placeholders. Note which files need editing (likely XML files under `Contents/`).

### 5. Examine the XML files containing placeholders
For each file found in step 4, read its full contents:
```bash
cat template_extracted/Contents/section0.xml
```
(Adjust path as needed based on grep results.)

### 6. Build and run a Python replacement script
Write a Python script that:
- Reads `supplier_contact.json`
- For each XML file containing placeholders, performs text replacement of every `{{KEY}}` with the corresponding JSON value
- **Critical**: After replacing text in any paragraph element, removes stale layout-cache elements. In HWPX XML, these are typically `<hp:linesegarray>` or `<lineseg>` or similar caching elements inside paragraph (`<hp:p>`) tags. Any element that caches glyph positions or line layout within a modified paragraph must be removed so the document renders cleanly.
- Writes the modified XML back

Here is the general approach for the script:
```python
import json
import os
import re
import shutil
from lxml import etree

# Load JSON
with open('/root/supplier_contact.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Build flat replacement map: handle nested JSON by flattening or by matching keys
# First inspect the JSON structure to determine key format

extracted = '/root/hwpx_work/template_extracted'

# Find all files with placeholders
for root, dirs, files in os.walk(extracted):
    for fname in files:
        fpath = os.path.join(root, fname)
        if not fname.endswith('.xml'):
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        if '{{' not in content:
            continue
        # Process this file...
```

The script must:
- Handle the exact placeholder format found in the XML (the placeholders might be split across XML elements — check for this)
- If placeholders are split across multiple XML text nodes within a single run/span, concatenate and re-split appropriately
- Replace ALL `{{...}}` patterns with corresponding JSON values
- Remove layout cache elements from modified paragraphs

### 7. Handle split placeholders
After step 5, carefully check whether any `{{...}}` placeholder is split across multiple XML elements (e.g., `{{` in one `<hp:t>` and `KEY}}` in another). If so, the script must handle merging text nodes before replacement. This is a common issue in word processor XML formats.

### 8. Remove stale layout-cache elements
In the XML, look for elements like `<hp:linesegarray>`, `<linesegarray>`, or any element that appears to cache line/character positioning within paragraph elements. After modifying a paragraph's text, delete these cache elements from that paragraph. Use namespace-aware XPath if needed.

### 9. Verify no placeholders remain
```bash
grep -r '{{' template_extracted/
```
This must return NO results. If any remain, fix the replacement logic.

### 10. Repackage the HWPX file
Repackage the modified files back into a ZIP with `.hwpx` extension:
```bash
cd /root/hwpx_work/template_extracted
zip -r /root/supplier_contact_ready.hwpx . -x '*.DS_Store'
```
IMPORTANT: The ZIP must be created from inside the extracted directory so paths are relative (no extra directory nesting). The mimetype file, if present, should ideally be stored first without compression (like in ODF/EPUB), but at minimum ensure the structure matches the original.

### 11. Validate the output
```bash
# Verify it's a valid ZIP
unzip -l /root/supplier_contact_ready.hwpx

# Verify no placeholders remain
unzip -p /root/supplier_contact_ready.hwpx | grep -c '{{'
# Should output 0

# Verify the file exists at the correct path
ls -la /root/supplier_contact_ready.hwpx
```

### 12. Final content verification
Extract and display the modified XML to confirm:
- All `{{...}}` placeholders are replaced with actual values from JSON
- Korean field labels are preserved
- Static note lines are unchanged
- Layout cache elements are removed from modified paragraphs

## Key Constraints
- Do NOT remove or alter Korean labels (e.g., '회사명:', '담당자:' etc.)
- Do NOT modify static/note lines that don't contain placeholders
- ALL `{{...}}` placeholders must be replaced — zero may remain
- The output must be a valid .hwpx (ZIP) package at `/root/supplier_contact_ready.hwpx`
- Remove layout-cache elements (like `<hp:linesegarray>` or similar) from any paragraph whose text content was modified

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