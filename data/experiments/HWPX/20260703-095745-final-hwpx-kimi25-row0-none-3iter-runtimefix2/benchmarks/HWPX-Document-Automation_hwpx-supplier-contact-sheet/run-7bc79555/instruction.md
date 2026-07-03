# Task Instruction

You are working with an HWPX file (a ZIP-based Korean word-processor format). Your goal: substitute every `{{…}}` placeholder in the template with the corresponding value from a JSON file, then save a clean result.

## Steps

### 1. Inspect the JSON data
```bash
cat /root/supplier_contact.json
```
Understand every key-value pair. These are the substitution values.

### 2. Explore the HWPX package
```bash
cd /root
mkdir -p hwpx_work
cp supplier_contact_template.hwpx hwpx_work/template.zip
cd hwpx_work
unzip -o template.zip -d template_contents
find template_contents -type f | sort
```
Identify all XML files, especially those under `Contents/` (e.g., `section0.xml`, `section1.xml`, etc.).

### 3. Find every placeholder occurrence
Search for `{{` across ALL files in the extracted package:
```bash
grep -rn '{{' template_contents/
```
Also search for partial placeholder fragments that may be split across XML tags:
```bash
grep -rn '\{\{\|\}\}' template_contents/
```
Note which files contain placeholders.

### 4. Examine the XML structure around placeholders
For each file containing placeholders, cat the full file content so you can see:
- Whether `{{key}}` appears intact in a single `<hp:t>` element, or
- Whether it is fragmented across multiple `<hp:t>` or `<hp:run>` elements (e.g., `<hp:t>{{</hp:t>`, `<hp:t>name</hp:t>`, `<hp:t>}}</hp:t>`).

This is critical. Do NOT assume placeholders are intact.

### 5. Write a Python replacement script
Write a Python script (`/root/hwpx_work/replace.py`) that:

1. Loads `supplier_contact.json`.
2. For each XML file that contained placeholders (from step 3):
   a. Reads the raw XML as a UTF-8 string.
   b. **First**, collapses fragmented placeholders: uses a regex to concatenate all `<hp:t>` text within each `<hp:run>` or `<hp:p>` block so that split `{{key}}` tokens become whole. Specifically:
      - Parse the XML with `lxml.etree` (or `xml.etree.ElementTree`).
      - For each paragraph element (`<hp:p>`), collect all `<hp:t>` text nodes in document order, join them, and check if the joined text contains any `{{...}}` pattern.
      - If a placeholder spans multiple `<hp:t>` nodes, consolidate the placeholder text into the first `<hp:t>` node and clear the others (set their `.text = ''`).
   c. **Then**, for each `<hp:t>` element, replace every `{{key}}` with the corresponding JSON value using simple string replacement. Ensure all JSON keys are tried.
   d. **Remove layout-cache elements**: For every `<hp:p>` paragraph whose text was modified, find and remove any child `<hp:lineSegArray>` element (and its descendants). This prevents overlapping-character rendering issues. Use the correct namespace when searching.
   e. Serialize the modified XML back to a UTF-8 string and write it back to the extracted file.

3. After processing all XML files, verify that NO `{{` remains in any file:
   ```python
   # scan all files for remaining placeholders
   ```
   If any remain, print a warning with the file and line.

### 6. Re-package the HWPX
Re-create the ZIP (HWPX) file from the modified contents, preserving the original directory structure and using deflate compression:
```python
import zipfile, os
output = '/root/supplier_contact_ready.hwpx'
base = '/root/hwpx_work/template_contents'
with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(base):
        for f in files:
            full = os.path.join(root, f)
            arcname = os.path.relpath(full, base)
            zf.write(full, arcname)
```

### 7. Validate the output
```bash
# Confirm it's a valid ZIP
unzip -t /root/supplier_contact_ready.hwpx

# Confirm no placeholders remain
unzip -p /root/supplier_contact_ready.hwpx | grep -c '{{'
# Expected: 0

# Spot-check: print text from section XML to confirm JSON values appear
unzip -p /root/supplier_contact_ready.hwpx Contents/section0.xml | grep -o '[^<>]*' | grep -i -E '(company|phone|email|fax|address)' | head -20
```

### 8. Final file location
The finished file MUST be at exactly `/root/supplier_contact_ready.hwpx`.

## Critical Reminders
- **Fragmented placeholders**: The #1 failure mode. Always consolidate split `<hp:t>` nodes before replacing.
- **Layout cache removal**: Always remove `<hp:lineSegArray>` from any modified `<hp:p>`. Use namespace-aware lookup.
- **Korean labels**: Do NOT remove or alter Korean text labels (like 회사명:, 전화:, etc.). Only replace `{{…}}` tokens.
- **Static note lines**: Leave any note/comment lines unchanged.
- **All keys**: Make sure every JSON key is used and every placeholder is replaced. Verify with grep at the end.
- **Namespace handling**: HWPX uses namespaces like `http://www.hancom.co.kr/hwpml/2011/paragraph` for `hp:`. Register them properly when using ElementTree/lxml.

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