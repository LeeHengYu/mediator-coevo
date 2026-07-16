# Task Instruction

Complete the clinic intake summary by filling a HWPX template with patient data. Follow these steps precisely:

## Step 1: Inspect the workspace
```bash
ls -la /root/
cat /root/patient_intake.json
```

## Step 2: Examine the HWPX template structure
HWPX files are ZIP archives containing XML files.
```bash
cd /root
python3 -c "import zipfile; z=zipfile.ZipFile('clinic_intake_template.hwpx','r'); print('\n'.join(z.namelist()))"
```

## Step 3: Extract and inspect all XML content for placeholders
Extract the HWPX to a temporary directory and search for all `{{` placeholders:
```bash
mkdir -p /tmp/hwpx_work
cd /tmp/hwpx_work
python3 -c "
import zipfile, os
z = zipfile.ZipFile('/root/clinic_intake_template.hwpx', 'r')
z.extractall('/tmp/hwpx_work')
z.close()
"
grep -r '{{' /tmp/hwpx_work/ --include='*.xml' -l
grep -rn '{{' /tmp/hwpx_work/ --include='*.xml'
```

Also search in ALL files (not just .xml) for placeholders:
```bash
grep -r '{{' /tmp/hwpx_work/ -l
```

## Step 4: Inspect the XML files containing placeholders
For each file found, read its full content to understand the XML structure, namespace declarations, and how text/placeholders are embedded. Pay special attention to:
- The exact placeholder syntax (e.g., `{{patient_name}}`, `{{birth_date}}`, etc.)
- Layout-cache elements near text paragraphs (elements like `linesegarray`, `lineSegArray`, `LineSeg`, or similar cached layout data)
- The XML namespace prefixes used

## Step 5: Write and run the Python transformation script
Create a Python script `/tmp/fill_template.py` that:

1. **Reads** `/root/patient_intake.json` to get all patient values.
2. **Opens** the template HWPX as a ZIP, iterates over every entry.
3. **For each XML file** (or any text-based file) in the archive:
   a. Decode as UTF-8.
   b. Find all `{{...}}` placeholders and replace them with corresponding JSON values.
   c. **Phone normalization**: For any phone/callback number placeholder, strip all non-digit characters from the value, then format as `NNN-NNNN-NNNN` (3-4-4 digit groups with hyphens). The field in the JSON may be named something like `callback_phone`, `phone`, `contact_number`, etc.
   d. **Age note**: After replacing the birth date placeholder, append ` (<N>세)` where N is the Korean full-year age (만 나이). Calculate as: parse the birth date and visit date, compute age = visit_year - birth_year, then subtract 1 if the visit date is before the birthday in that year. The age note goes right after the birth date text in the same text run/element.
   e. **Patient name confirmation**: Ensure ALL occurrences of the patient name placeholder are replaced, including any repeated confirmation line.
   f. **Remove stale layout-cache elements**: For any paragraph element whose text content was modified, remove child elements that represent layout caches. These are typically elements with local names like `linesegarray`, `lineSegArray`, `LineSeg`, `lineseg`, or similar. Use an XML parser (lxml or xml.etree.ElementTree) for this step rather than regex, to safely identify and remove these elements by their tag name. Look for elements containing 'LineSeg' or 'lineseg' (case-insensitive) in their tag name within modified paragraphs.
4. **Writes** the result to `/root/clinic_intake_ready.hwpx` as a valid ZIP, preserving the original compression method and directory structure.

IMPORTANT implementation notes:
- Use `lxml` or `xml.etree.ElementTree` for XML parsing when removing layout-cache elements. For placeholder replacement, you may use string replacement on the raw XML text first, then parse the result to remove layout caches from modified paragraphs.
- Alternatively, do everything with an XML parser: parse each XML file, walk the tree to find text nodes containing `{{...}}`, replace them, mark their parent paragraphs, then remove layout-cache children from those paragraphs.
- When writing the output ZIP, preserve the exact same entry names. Do NOT add extra directory entries or change the structure.
- For non-XML binary entries (images, etc.), copy them as-is.

## Step 6: Run the script
```bash
python3 /tmp/fill_template.py
```

## Step 7: Validate the output
```bash
# Verify it's a valid ZIP/HWPX
python3 -c "import zipfile; z=zipfile.ZipFile('/root/clinic_intake_ready.hwpx','r'); print('Valid ZIP. Entries:'); print('\n'.join(z.namelist())); z.close()"

# Verify NO placeholders remain anywhere
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/clinic_intake_ready.hwpx', 'r')
found = False
for name in z.namelist():
    try:
        content = z.read(name).decode('utf-8', errors='ignore')
        import re
        matches = re.findall(r'\{\{[^}]*\}\}', content)
        if matches:
            print(f'REMAINING PLACEHOLDERS in {name}: {matches}')
            found = True
    except: pass
if not found:
    print('OK: No placeholders remain.')
z.close()
"

# Verify age note is present
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/clinic_intake_ready.hwpx', 'r')
for name in z.namelist():
    try:
        content = z.read(name).decode('utf-8', errors='ignore')
        if '세)' in content:
            import re
            m = re.findall(r'\(\d+세\)', content)
            if m: print(f'Age note found in {name}: {m}')
    except: pass
z.close()
"

# Verify phone format
python3 -c "
import zipfile, re
z = zipfile.ZipFile('/root/clinic_intake_ready.hwpx', 'r')
for name in z.namelist():
    try:
        content = z.read(name).decode('utf-8', errors='ignore')
        phones = re.findall(r'\d{3}-\d{4}-\d{4}', content)
        if phones: print(f'Phone in {name}: {phones}')
    except: pass
z.close()
"

# Show the text content of the main document XML for visual inspection
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/clinic_intake_ready.hwpx', 'r')
for name in z.namelist():
    if 'section' in name.lower() or 'content' in name.lower():
        print(f'=== {name} ===')
        print(z.read(name).decode('utf-8')[:3000])
z.close()
"
```

If any validation fails, debug and fix the script, then re-run.

## Step 8: Verify layout-cache removal
Confirm that modified paragraphs do not contain stale layout-cache elements:
```bash
python3 -c "
import zipfile, re
z = zipfile.ZipFile('/root/clinic_intake_ready.hwpx', 'r')
for name in z.namelist():
    try:
        content = z.read(name).decode('utf-8', errors='ignore')
        # Check for common layout cache element patterns
        caches = re.findall(r'<[^>]*[Ll]ine[Ss]eg[^>]*>', content)
        if caches:
            print(f'Layout cache elements in {name}: {len(caches)} found')
            # This is only a problem if they're in paragraphs we modified
    except: pass
z.close()
"
```

The final output must be at `/root/clinic_intake_ready.hwpx`.

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