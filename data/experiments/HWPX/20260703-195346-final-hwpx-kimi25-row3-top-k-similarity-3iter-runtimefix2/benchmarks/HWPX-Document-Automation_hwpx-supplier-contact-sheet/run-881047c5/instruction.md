# Task Instruction

Perform the following steps to produce `/root/supplier_contact_ready.hwpx`.

## 1. Reconnaissance

1. `ls /root/` — find the template and JSON files.
2. Read `supplier_contact.json` fully.
3. Unzip `supplier_contact_template.hwpx` into a temp directory (e.g., `/tmp/hwpx_work/`) and list all files.
4. Identify which XML file(s) inside the HWPX contain `{{` placeholders. Typically this is `Contents/section0.xml` but verify. Print the full content of each XML file that contains `{{`.
5. Also print the `mimetype` file content and note its exact bytes.
6. Inspect the XML for namespace declarations — note the exact namespace URI for `hp:` prefix.
7. Look for any `lineSegArray` or `linesegarray` tags (case-insensitive search) in the XML to understand the exact tag name used. Run: `grep -ri 'lineseg' /tmp/hwpx_work/`

## 2. Examine the test/verifier

Look for any test or verifier script in the task directory:
```
find /root/ -name '*.py' -o -name 'verify*' -o -name 'test*' | head -20
```
Read any test file found to understand exact validation expectations (placeholder check, layout cache check, valid ZIP check, field labels, static note line, etc.).

## 3. Write and run the transformation script

Create `/root/transform.py` that does the following:

### a. Register namespaces
Before any XML parsing, register ALL namespace prefixes found in the XML so that `ElementTree` serialization preserves them exactly. Use `ET.register_namespace(prefix, uri)` for every namespace.

### b. Load JSON
Read `supplier_contact.json` into a dict.

### c. Parse the XML
Parse the section XML file with `xml.etree.ElementTree`.

### d. Replace placeholders
Iterate over all `<hp:t>` text elements. For each element whose `.text` contains `{{...}}`:
- Use `re.sub` to replace every `{{key}}` with the corresponding value from the JSON dict.
- Track the parent `<hp:p>` paragraph element of every modified `<hp:t>` node.

### e. Remove ALL layout cache elements from modified paragraphs
This is the critical step that failed before. Do NOT hardcode a single tag name. Instead, for every modified paragraph element, search for **both** possible tag forms:
- `{namespace}lineSegArray`
- `{namespace}linesegarray`

Also do a broader sweep: iterate over all child/descendant elements of each modified `<hp:p>` and remove any element whose local tag name (case-insensitive) matches `linesegarray`. Code pattern:
```python
for p_elem in modified_paragraphs:
    for child in list(p_elem.iter()):
        local = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        if local.lower() == 'linesegarray':
            parent = child.getparent() if hasattr(child, 'getparent') else None
            # For stdlib ElementTree, we need parent map
            # Use a parent_map built earlier
            if child in parent_map:
                parent_map[child].remove(child)
```
Build a `parent_map` via `{c: p for p in root.iter() for c in p}` before the removal loop.

Alternatively, use `lxml.etree` if available (check with `python3 -c 'import lxml'`), which supports `.getparent()` natively.

### f. Verify no placeholders remain
After replacement, scan all text in the XML tree for `{{`. If any remain, raise an error listing them.

### g. Write the XML back
Serialize with `xml_declaration=True, encoding='UTF-8'`.

### h. Repackage the HWPX ZIP
Create `/root/supplier_contact_ready.hwpx` as a ZIP file:
- First entry must be `mimetype` stored **uncompressed** (`compression=ZIP_STORED`).
- All other files use `ZIP_DEFLATED`.
- Walk the temp directory and add every file with the correct archive path.

## 4. Run and verify

1. Run: `python3 /root/transform.py`
2. Verify the output:
   - `python3 -c "import zipfile; z=zipfile.ZipFile('/root/supplier_contact_ready.hwpx'); print(z.namelist()); print(z.read('mimetype'))"` — confirm valid ZIP, mimetype first.
   - `python3 -c "import zipfile; z=zipfile.ZipFile('/root/supplier_contact_ready.hwpx'); [print(f) for f in z.namelist() if 'section' in f]; import xml.etree.ElementTree as ET; ..."` — extract section XML and grep for `{{` to confirm none remain.
   - Search the output section XML for `linesegarray` (case-insensitive) in modified paragraphs to confirm removal.
3. If any test script was found in step 2, run it to confirm all checks pass.
4. If anything fails, diagnose and fix before declaring done.

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