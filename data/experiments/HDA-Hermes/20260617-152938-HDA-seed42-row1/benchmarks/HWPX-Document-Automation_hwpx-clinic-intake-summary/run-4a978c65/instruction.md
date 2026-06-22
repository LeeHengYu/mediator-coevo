# Task Instruction

Complete the clinic intake summary HWPX document by filling in all placeholders from the patient data JSON and saving the result.

## Steps

### 1. Inspect the workspace
```bash
ls /root/
find /root/ -name '*.hwpx' -o -name '*.json' | head -20
```

### 2. Examine the patient data
```bash
cat /root/patient_intake.json
```
Note all field values. Pay special attention to:
- Patient name (will appear multiple times including a confirmation line)
- Birth date and visit date (needed for age calculation)
- Phone number (needs normalization to `000-0000-0000` format)

### 3. Examine the HWPX template structure
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('/root/clinic_intake_template.hwpx', 'r') as z:
    for name in z.namelist():
        print(name)
"
```

### 4. Find and inspect the main content XML
The main content is typically in `Contents/section0.xml` (or similar). Extract and inspect it:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('/root/clinic_intake_template.hwpx', 'r') as z:
    for name in z.namelist():
        if 'section' in name.lower() or 'content' in name.lower():
            print(f'--- {name} ---')
            print(z.read(name).decode('utf-8')[:5000])
"
```
Identify all `{{...}}` placeholders and the XML namespace prefixes used.

### 5. Write and run the transformation script

Create a Python script `/root/fill_template.py` that does the following:

```python
import json, zipfile, shutil, re, copy, os
from lxml import etree
from datetime import date

# Load patient data
with open('/root/patient_intake.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# --- Compute derived values ---

# Korean full-year age (만 나이): age as of visit date
# Parse dates from the JSON (inspect actual format first)
# Example: if birth_date is "1985-03-15" and visit_date is "2024-01-10"
# Full-year age = visit_year - birth_year - (1 if birthday hasn't occurred yet this year else 0)

# Normalize phone number: strip all non-digits, format as 000-0000-0000
# e.g., "010 1234 5678" -> "010-1234-5678"

# --- Build replacement map ---
# Map each placeholder key to its replacement value
# Include the age note: after birth date, append " (<N>세)"

# --- Process HWPX ---
template_path = '/root/clinic_intake_template.hwpx'
output_path = '/root/clinic_intake_ready.hwpx'

# Copy template to output
shutil.copy2(template_path, output_path)

# Open the output HWPX (it's a zip)
with zipfile.ZipFile(template_path, 'r') as zin:
    with zipfile.ZipFile(output_path, 'w') as zout:
        for item in zin.namelist():
            raw = zin.read(item)
            if item.endswith('.xml'):
                # Check if this XML contains any placeholders
                text = raw.decode('utf-8')
                if '{{' in text or '}}' in text:
                    # Parse with lxml, handle namespaces
                    root = etree.fromstring(raw)
                    # Process paragraphs to handle split placeholders
                    # ... (see detailed logic below)
                    raw = etree.tostring(root, xml_declaration=True, encoding='utf-8')
            zout.writestr(item, raw)
```

**Critical implementation details for the script:**

a) **Split placeholder handling**: Placeholders like `{{patient_name}}` may be split across multiple `<hp:t>` (or similar text) elements within a single paragraph `<hp:p>`. To handle this:
   - For each paragraph element, collect all text-bearing child elements (the `<hp:t>` or equivalent nodes that contain the actual text runs).
   - Concatenate all their text content into a single string.
   - Perform all `{{...}}` replacements on the concatenated string.
   - Put the entire replaced text into the first text node and clear the rest.

b) **Layout cache removal**: For any paragraph where text was modified, find and remove `<hp:lineSegArray>` (or any `lineSegArray`/`lineSeg` elements) within that paragraph. Use the correct namespace. This prevents overlapping character rendering.

c) **Age calculation**: Compute Korean 만 나이 (full-year age):
   ```python
   birth = date(...)  # from JSON
   visit = date(...)  # from JSON  
   age = visit.year - birth.year - ((visit.month, visit.day) < (birth.month, birth.day))
   ```
   The age note `(<N>세)` should be appended after the birth date value in the document, with a space before the parenthesis.

d) **Phone normalization**: Strip all non-digit characters, then format as `XXX-XXXX-XXXX`:
   ```python
   digits = re.sub(r'\D', '', phone_raw)
   phone = f"{digits[:3]}-{digits[3:7]}-{digits[7:11]}"
   ```

e) **Namespace handling**: Parse the XML to discover the namespace URIs (e.g., `http://www.hancom.co.kr/hwpml/2011/paragraph` for `hp`). Use these in all XPath queries and element lookups.

f) **Preserve Korean labels and handwritten-signature note**: Only replace `{{...}}` placeholders. Do not modify any other text content.

### 6. Run the script
```bash
python3 /root/fill_template.py
```

### 7. Validate the output

Run these validation checks:

```bash
python3 -c "
import zipfile

# Check it's a valid zip/hwpx
path = '/root/clinic_intake_ready.hwpx'
with zipfile.ZipFile(path, 'r') as z:
    print('Valid zip. Files:', len(z.namelist()))
    for name in z.namelist():
        print(' ', name)
        raw = z.read(name)
        if name.endswith('.xml'):
            text = raw.decode('utf-8')
            # Check no remaining placeholders
            import re
            matches = re.findall(r'\{\{.*?\}\}', text)
            if matches:
                print(f'  WARNING: remaining placeholders in {name}: {matches}')
            # Check for lineSegArray in modified content
            # (informational)
print('Validation complete.')
"
```

Also verify specific content:
```bash
python3 -c "
import zipfile, re
path = '/root/clinic_intake_ready.hwpx'
with zipfile.ZipFile(path, 'r') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            text = z.read(name).decode('utf-8')
            if '세)' in text:
                print(f'Age note found in {name}')
            # Check phone format
            phones = re.findall(r'\d{3}-\d{4}-\d{4}', text)
            if phones:
                print(f'Phone numbers in {name}: {phones}')
            # Check no {{ or }} remain
            if '{{' in text or '}}' in text:
                print(f'ERROR: Placeholders remain in {name}')
"
```

### 8. Run the test suite if present
```bash
cd /root && ls test_output.py 2>/dev/null && python3 -m pytest test_output.py -v
```

If any test fails, read the error message, fix the issue in the script, re-run, and re-validate.

### Key constraints to remember:
- Output file MUST be at `/root/clinic_intake_ready.hwpx` (check the avoid artifact - a prior task failed because the output wasn't created)
- No `{{...}}` placeholders may remain anywhere in any XML file in the package
- The age note format is exactly `(<N>세)` with parentheses
- Phone format is exactly `000-0000-0000` (3-4-4 digit groups with hyphens)
- Layout cache (`lineSegArray` elements) must be removed from modified paragraphs
- The file must be a valid HWPX (zip) package with all original entries preserved

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