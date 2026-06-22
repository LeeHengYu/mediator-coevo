# Task Instruction

Execute the following steps in order to produce `/root/clinic_intake_ready.hwpx`.

## 1 – Inspect the workspace
```bash
ls /root/
cat /root/patient_intake.json
```
Understand every key in the JSON (patient name, birth date, visit date, phone, address, symptoms, etc.).

## 2 – Explore the template HWPX
```bash
cd /root
python3 -c "
import zipfile, os
with zipfile.ZipFile('clinic_intake_template.hwpx') as z:
    for info in z.infolist():
        print(info.filename, info.compress_type, info.file_size)
"
```
Then extract and print every `section*.xml` file:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('clinic_intake_template.hwpx') as z:
    for name in z.namelist():
        if 'section' in name.lower() and name.endswith('.xml'):
            print('===', name, '===')
            print(z.read(name).decode('utf-8'))
"
```
Also print any other XML files (header, content.hpf, manifest, etc.) that might contain placeholder text:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('clinic_intake_template.hwpx') as z:
    for name in z.namelist():
        if name.endswith('.xml') or name.endswith('.hpf') or name == 'mimetype':
            data = z.read(name)
            if b'{{' in data or name == 'mimetype':
                print('===', name, '===')
                print(data.decode('utf-8'))
"
```
Record every `{{PLACEHOLDER}}` you find and which file(s) contain them.

## 3 – Understand the verifier
```bash
cat /root/test_output.py
```
Read the test file carefully. Note every assertion, expected string, and structural check. This is the contract you must satisfy.

## 4 – Write and run the generation script
Create `/root/solve.py` that does the following:

### 4a – Load data
- Read `patient_intake.json`.
- Read the template HWPX as a ZIP.

### 4b – Compute derived values
- **Korean full-year age**: From the patient's birth date and the visit date, compute `visit_year - birth_year`. If the patient has not yet had their birthday in the visit year (month/day comparison), subtract 1. Format as `(<N>세)`. This note must appear immediately after the birth-date value in the same paragraph or the same text run, separated by a space.
- **Phone normalization**: Strip all non-digit characters from the callback phone number, then format as `NNN-NNNN-NNNN` (3-4-4 grouping for an 11-digit Korean mobile number). If the number has 10 digits, use `NNN-NNN-NNNN` (3-3-4). Prefer 3-4-4 for 11 digits.

### 4c – Placeholder replacement in XML
For every XML file in the archive that contains `{{`:
1. Parse with `xml.etree.ElementTree`.
2. Register all namespaces found in the file (use `xml.etree.ElementTree.iterparse` with `'start-ns'` events, then `ET.register_namespace` for each) so they are preserved on write.
3. **Reassemble split placeholders**: HWPX often splits `{{...}}` across multiple `<hp:t>` (or similar text) elements within a paragraph (`<hp:p>`). For each paragraph element:
   a. Collect all text-bearing child elements (recursively find elements whose `.text` contains partial placeholder text, or more robustly, iterate `<hp:run>` children).
   b. Concatenate their `.text` values to form the full paragraph text.
   c. If the concatenated text contains any `{{...}}` pattern, perform all replacements on the concatenated string.
   d. Place the entire replaced text into the first text element's `.text`, and set all subsequent text elements' `.text` to `""`.
   e. Alternatively, if the structure is simpler (each placeholder is in one `<hp:t>`), a simple find-and-replace on each element's `.text` is sufficient. **Check both `.text` and `.tail` of every element.**
4. After replacing text in a paragraph, **remove any `<hp:lineSegArray>` child** (or any element whose local name is `lineSegArray` or `linesegarray`) from that paragraph to clear the layout cache.
5. Also remove `<hp:lineSegArray>` from any ancestor `<hp:p>` of the modified element.

### 4d – Build the replacement map
Map each `{{PLACEHOLDER}}` to the correct JSON value. Be sure to:
- Replace `{{patient_name}}` (or whatever the exact placeholder names are) everywhere, including any confirmation line.
- Insert the age note after the birth-date value: e.g., `1985-03-15 (39세)` (compute the actual age).
- Use the normalized phone number.
- Preserve all Korean labels and the handwritten-signature note verbatim.

### 4e – Verify no remaining placeholders
After all replacements, scan every XML file's serialized text for `{{`. If any remain, raise an error listing them.

### 4f – Write the output HWPX
Write `/root/clinic_intake_ready.hwpx` as a ZIP:
1. The `mimetype` entry must be **first** in the archive and stored with **compression method 0** (stored, no compression).
2. All other entries use their original compression method (typically deflated).
3. Preserve all non-XML files (images, etc.) byte-for-byte from the original.
4. For modified XML files, write the re-serialized XML (with `xml_declaration=True, encoding='utf-8'`).

### Run it
```bash
python3 /root/solve.py
```

## 5 – Validate
```bash
# Check it's a valid ZIP / HWPX
python3 -c "
import zipfile
with zipfile.ZipFile('/root/clinic_intake_ready.hwpx') as z:
    for info in z.infolist():
        print(info.filename, info.compress_type)
    # Verify mimetype is first and stored
    assert z.infolist()[0].filename == 'mimetype'
    assert z.infolist()[0].compress_type == 0
    print('ZIP structure OK')
"

# Check no remaining placeholders
python3 -c "
import zipfile
with zipfile.ZipFile('/root/clinic_intake_ready.hwpx') as z:
    for name in z.namelist():
        data = z.read(name)
        if b'{{' in data:
            print('FAIL: placeholder remains in', name)
            print(data.decode('utf-8', errors='replace'))
        else:
            print('OK:', name)
"

# Run the actual verifier
cd /root && python3 -m pytest test_output.py -v
```

## 6 – Fix and iterate
If any test fails, read the error message carefully, re-inspect the relevant XML in the output HWPX, compare with the verifier expectations, fix `solve.py`, and re-run. Common issues:
- Placeholder name mismatch (case, underscores, spaces)
- Age calculation off-by-one
- Phone format wrong
- Split-run placeholders not reassembled
- Layout cache elements not removed
- Namespace prefixes lost during serialization
- Birth-date age note not in the exact expected format or position

Repeat until `pytest` passes all tests with no failures.

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