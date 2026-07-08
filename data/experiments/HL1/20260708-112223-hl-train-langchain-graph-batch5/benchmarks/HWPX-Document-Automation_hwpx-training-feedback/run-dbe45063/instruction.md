# Task Instruction

Complete the following task step by step.

## Goal
Fill in the training feedback sheet `training_feedback_template.hwpx` using values from `training_feedback.json`, then save the result to `/root/training_feedback_ready.hwpx`.

## Steps

### Step 1: Examine the workspace
```bash
ls -la /root/
find /root/ -name '*.hwpx' -o -name '*.json' | head -20
```
Identify the exact paths of `training_feedback_template.hwpx` and `training_feedback.json`.

### Step 2: Read the JSON data
```bash
cat <path_to>/training_feedback.json
```
Note every key-value pair. You will need to map these to `{{...}}` placeholders.

### Step 3: Explore the HWPX package structure
HWPX files are ZIP archives. Unzip to a temporary directory:
```bash
mkdir -p /tmp/hwpx_work
cp <path_to>/training_feedback_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_extracted
find template_extracted -type f
```
List all files. The main content is typically in `Contents/section0.xml` (or similar XML files under `Contents/`). There may also be a `header.xml` or other XML files.

### Step 4: Search for ALL placeholders
```bash
grep -rn '{{' template_extracted/
```
This will show every file and line containing `{{...}}` placeholders. Document ALL of them — you must replace every single one.

### Step 5: Understand the XML structure around placeholders
For each file containing placeholders, read the full content:
```bash
cat template_extracted/Contents/section0.xml
```
(and any other files with placeholders)

Pay special attention to:
- How text is split across XML elements (placeholders may span multiple `<hp:t>` tags or be within a single tag)
- Layout cache elements like `<hp:linesegarray>`, `<hp:lineseg>`, or `<hp:charshapeidarray>` near text runs — these are stale caches that must be removed from any paragraph you modify

### Step 6: Write a Python script to perform all replacements
Create `/tmp/hwpx_work/fill_template.py` that does the following:

```python
import json, os, shutil, zipfile, re
from lxml import etree

# 1. Load JSON
with open('<path_to>/training_feedback.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. Build replacement map from JSON keys to placeholder names
#    Map each {{key}} to its replacement value
#    Apply special formatting rules:
#    a) 참석자수: convert to digits only (e.g., "32명" -> "32", or if already numeric, keep as-is)
#    b) 만족도: rewrite as "X.X점 (5.0점 만점)" using the numeric score from JSON
#    c) For the overall opinion field: append " 후속 심화반 검토 요망." after the JSON value

# 3. For each XML file in the extracted HWPX:
#    a) Parse with lxml
#    b) Find all text nodes containing {{...}}
#    c) Replace placeholders with values
#    d) For any <hp:p> (paragraph) element that was modified, remove child elements
#       that are layout caches. These typically have tag names like:
#       - linesegarray
#       - Any element whose local name contains 'lineseg' or 'cache'
#       Look at the actual XML namespace and tag names in the file.
#    e) Write back the XML

# 4. Repack as HWPX (ZIP)
```

**CRITICAL RULES for the script:**

- **Placeholder spanning**: Placeholders like `{{교육명}}` might be split across multiple `<hp:t>` elements (e.g., `<hp:t>{{교육</hp:t><hp:t>명}}</hp:t>`). You must handle this. One robust approach: for each paragraph element, concatenate all text content, perform replacements on the concatenated string, then redistribute or place all text in a single `<hp:t>` element (while preserving the first run's formatting).

- **참석자수 formatting**: Extract only digits from the JSON value. For example, if the value is `"32명"`, output `"32"`. If it's already `32` (integer), convert to string `"32"`.

- **만족도 formatting**: If JSON has e.g. `4.5` or `"4.5"`, output `"4.5점 (5.0점 만점)"`.

- **Overall opinion**: Find the placeholder for the overall opinion/comment field. After substituting the JSON value, append ` 후속 심화반 검토 요망.` (with a space before 후속). Make sure the final text reads naturally.

- **Layout cache removal**: After modifying any paragraph's text, find and remove stale layout-cache child elements from that paragraph. Inspect the actual XML to identify the correct element names and namespaces. Common candidates: elements with local names like `linesegarray`, `lineseg`, or similar. Remove them entirely from modified paragraphs.

- **Korean labels and static note line**: Do NOT modify any text that is not a placeholder. Only replace `{{...}}` patterns.

- **Validation**: After all replacements, scan all XML files to confirm zero `{{` or `}}` patterns remain.

- **Repacking**: When creating the ZIP, preserve the original directory structure exactly. Use `ZIP_DEFLATED` compression. Make sure `mimetype` file (if present) is stored first and uncompressed (this is standard for OPC-like packages).

### Step 7: Run the script
```bash
cd /tmp/hwpx_work
python3 fill_template.py
```
Fix any errors.

### Step 8: Validate the output
```bash
# Check it's a valid ZIP
unzip -t /root/training_feedback_ready.hwpx

# Extract and verify no placeholders remain
mkdir -p /tmp/hwpx_verify
unzip /root/training_feedback_ready.hwpx -d /tmp/hwpx_verify
grep -rn '{{' /tmp/hwpx_verify/

# Verify specific values were correctly inserted
# Check 참석자수 is digits only
# Check 만족도 format
# Check 후속 심화반 검토 요망 appears
grep -rn '참석자수\|만족도\|후속 심화반' /tmp/hwpx_verify/

# Show the content XML for manual review
cat /tmp/hwpx_verify/Contents/section0.xml
```

### Step 9: Final checks
- Confirm `/root/training_feedback_ready.hwpx` exists and is non-empty
- Confirm it's a valid ZIP archive
- Confirm zero `{{...}}` placeholders remain in any XML file
- Confirm all Korean labels are preserved
- Confirm layout cache elements are removed from modified paragraphs

## Important Notes
- Read every file carefully before editing. The HWPX XML namespace and element names may differ from what you expect.
- If placeholders are split across XML elements, handle the merging carefully.
- The `lxml` library should be available. If not, use `xml.etree.ElementTree` but be careful with namespace handling.
- Do NOT use string replacement on raw XML — use proper XML parsing to avoid breaking tags.
- If you encounter unexpected XML structure, inspect it thoroughly before proceeding.

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