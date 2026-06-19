# Task Instruction

Execute the following Python script to produce `/root/safety_audit_brief_final.hwpx`.

```python
import json, os, re, shutil, zipfile
from copy import deepcopy
from xml.etree import ElementTree as ET

# ── paths ──
BASE = '/root'
TEMPLATE = os.path.join(BASE, 'safety_audit_template.hwpx')
OVERVIEW = os.path.join(BASE, 'audit_overview.json')
CORRECTIVE = os.path.join(BASE, 'corrective_actions.json')
OUTPUT = os.path.join(BASE, 'safety_audit_brief_final.hwpx')
WORK = os.path.join(BASE, '_hwpx_work')

# ── load JSON data ──
with open(OVERVIEW, 'r', encoding='utf-8') as f:
    overview = json.load(f)
with open(CORRECTIVE, 'r', encoding='utf-8') as f:
    corrective = json.load(f)

print('=== overview ===')
print(json.dumps(overview, indent=2, ensure_ascii=False))
print('=== corrective_actions ===')
print(json.dumps(corrective, indent=2, ensure_ascii=False))

# ── extract template ──
if os.path.isdir(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK)
with zipfile.ZipFile(TEMPLATE, 'r') as zf:
    zf.extractall(WORK)

# ── find all XML files inside the package ──
xml_files = []
for root, dirs, files in os.walk(WORK):
    for fn in files:
        if fn.endswith('.xml'):
            xml_files.append(os.path.join(root, fn))
print('XML files:', xml_files)

# ── Print contents of section*.xml or content*.xml to understand structure ──
for xf in xml_files:
    basename = os.path.basename(xf).lower()
    if 'section' in basename or 'content' in basename:
        with open(xf, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f'\n=== {xf} (first 8000 chars) ===')
        print(content[:8000])
```

After running this first script, inspect the output carefully to understand:
1. The structure of the JSON data files (field names, values, risk tier, inspection date format).
2. The XML structure of the HWPX template (namespace prefixes, placeholder patterns like `{{...}}`, section titles, table structure).
3. How `<hp:lineSegArray>` elements appear in the XML.

Then write and run a second comprehensive Python script that:

### Step A – Build replacement map
- From `audit_overview.json`, map each `{{placeholder}}` to its value. For the inspection date field, rewrite from `YYYY-MM-DD` to `YYYY.MM.DD`.
- Determine the risk tier value. Append the severity note using the mapping: `High -> 즉시조치`, `Medium -> 계획보완`, `Low -> 모니터링`. For example, if risk tier is `High`, every occurrence should become `High (즉시조치)` (or however the placeholder is structured — adapt after inspecting the template).
- From `corrective_actions.json`, map the three corrective-action placeholders in order.

### Step B – Process each XML file
For every `.xml` file in the extracted HWPX:
1. Parse the raw XML text.
2. Register all namespaces found in the file (use `re.findall` on `xmlns:` declarations and call `ET.register_namespace` for each) so they are preserved on serialization.
3. Parse with `ET.parse()`.
4. For every `<hp:p>` paragraph element (find with the hp namespace URI):
   a. Collect all `<hp:t>` text nodes within the paragraph.
   b. Concatenate their `.text` values to form the full paragraph text.
   c. Check if the concatenated text contains any `{{` placeholder or needs date reformatting or risk-tier update.
   d. If modifications are needed:
      - Perform all placeholder substitutions on the concatenated text.
      - Rewrite all dates from `YYYY-MM-DD` to `YYYY.MM.DD`.
      - Ensure no `{{...}}` remains.
      - Put the entire resulting text into the **first** `<hp:t>` element's `.text`, and set all subsequent `<hp:t>` elements' `.text` to empty string `''`.
      - **Remove all `<hp:lineSegArray>` elements** from this `<hp:p>` element (find them with proper namespace and use `parent.remove(child)` pattern — iterate through all descendants).
5. Also do a second pass: find and remove ALL `<hp:lineSegArray>` elements from paragraphs that contain any modified text. To be safe, after all placeholder work is done, do a global pass removing `<hp:lineSegArray>` from every `<hp:p>` that was touched.
6. Serialize back to the file using `ET.ElementTree(root).write(filepath, encoding='utf-8', xml_declaration=True)`.

### Step C – Also do a raw-text safety pass
After the XML-parser pass, re-read each modified XML file as raw text and:
- Verify no `{{` or `}}` remains. If any do, replace them with regex.
- Verify all dates are in `YYYY.MM.DD` format (no `YYYY-MM-DD` of the original date).
- As a belt-and-suspenders measure, remove any remaining `<hp:lineSegArray...>...</hp:lineSegArray>` using regex with `re.DOTALL`. Also handle the case where the namespace prefix might be different or where there's a default namespace (e.g., `<lineSegArray` without prefix). Print warnings if any are found and removed.
- Write back.

### Step D – Repackage as HWPX
- Create the output ZIP file.
- Add the `mimetype` file FIRST with `compression=zipfile.ZIP_STORED` (no compression).
- Add all other files with `compression=zipfile.ZIP_DEFLATED`.
- The result is `/root/safety_audit_brief_final.hwpx`.

### Step E – Validate
- Open the output HWPX as a ZIP and list its contents.
- Read the section XML(s) and confirm:
  - No `{{` or `}}` anywhere.
  - No `<hp:lineSegArray` or `<lineSegArray` anywhere in paragraphs that were modified.
  - The date appears in `YYYY.MM.DD` format.
  - The risk tier has the severity note appended.
  - The corrective actions appear in order.
- Print all validation results.

Clean up the temporary working directory after successful completion.

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