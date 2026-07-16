# Task Instruction

Complete the clinic intake summary HWPX document by filling in patient data from JSON. Follow these steps precisely:

## Step 1: Inspect the input files

1. Read `/root/patient_intake.json` to understand all available patient data fields.
2. Examine the HWPX template structure:
   ```
   cd /root
   python3 -c "
import zipfile
with zipfile.ZipFile('clinic_intake_template.hwpx', 'r') as z:
    for name in z.namelist():
        print(name)
   "
   ```
3. Extract and read each XML file in the package to find all `{{...}}` placeholders. Pay special attention to section XML files (likely under `Contents/` directory). Print the full content of each XML file that contains `{{`.

## Step 2: Understand the data and compute derived values

1. Parse the JSON to get all patient fields.
2. Compute Korean full-year age: This is `visit_year - birth_year`, adjusted down by 1 if the visit date is before the birthday in that year. Format as `(<N>세)` where N is the integer age.
3. Normalize the callback phone number: strip all non-digit characters, then format as `NNN-NNNN-NNNN` (Korean mobile format: 3-4-4 digit groups with hyphens).

## Step 3: Build the replacement mapping

Create a Python dictionary mapping every `{{placeholder_name}}` found in the XML to its replacement value from the JSON. Include:
- All direct field mappings (patient name, birth date, visit date, address, etc.)
- The age note: after replacing the birth date placeholder, append ` (<N>세)` to the birth date value
- The normalized phone number
- Any repeated placeholders (e.g., patient name may appear multiple times including a confirmation line)

## Step 4: Write the transformation script

Write a Python script `/root/fill_template.py` that:

```python
import zipfile
import json
import re
import os
import shutil
from datetime import date
import xml.etree.ElementTree as ET

# 1. Load patient data
with open('patient_intake.json', 'r', encoding='utf-8') as f:
    patient = json.load(f)

# 2. Compute age and normalize phone
# ... (compute Korean full-year age, format phone as NNN-NNNN-NNNN)

# 3. Build replacement map from {{key}} -> value
# Map every placeholder found in the template to the correct value

# 4. Read the HWPX (ZIP) file
# For each file in the ZIP:
#   - If it's an XML file containing {{...}} placeholders:
#     a. Perform all text replacements
#     b. Parse the modified XML
#     c. For any <hp:p> paragraph element (or equivalent) whose text was modified,
#        remove layout-cache child elements. These typically include elements like:
#        - linesegarray / LINESEGARRAY
#        - charShapeIds / CHARSHAPEIDS  
#        - Any element related to cached layout/positioning
#        Look for elements with local names containing 'lineseg', 'charShapeId',
#        or similar layout-cache patterns within paragraph elements.
#     d. Serialize back to XML string preserving encoding and declarations
#   - Copy all other files unchanged

# 5. Write to /root/clinic_intake_ready.hwpx
```

IMPORTANT implementation details:

### Layout cache removal
- After replacing text in an XML file, you MUST identify which paragraphs were modified and remove their layout-cache elements.
- Parse the XML with namespace awareness. Inspect the actual namespace URIs used in the document.
- Common HWPX layout-cache elements within `<hp:p>` paragraphs include tags with local names like: `linesegarray`, `lineSegArray`, `charShapeIds`. Inspect the actual XML to determine the exact element names and namespaces.
- Strategy: Before replacement, record which paragraph elements contain placeholder text. After replacement, remove layout-cache children from those paragraphs.
- Alternative simpler strategy: Do text replacement on the raw XML string, then parse it, find all paragraph elements, check if they differ from the original, and strip layout-cache elements from changed ones.

### XML handling
- Be very careful with XML namespaces. Register all namespaces before parsing to avoid namespace prefix changes in output.
- Preserve XML declarations, encoding attributes, and any processing instructions.
- When serializing back, ensure the output matches the original encoding.

### ZIP handling
- Preserve the exact same file list and compression settings.
- Use `zipfile.ZIP_DEFLATED` or match the original compression method.
- Do NOT include extra files or directories.

## Step 5: Execute and validate

1. Run the script: `python3 /root/fill_template.py`
2. Verify the output exists: `ls -la /root/clinic_intake_ready.hwpx`
3. Verify it's a valid ZIP: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/clinic_intake_ready.hwpx'); print(z.namelist()); z.close()"`
4. Verify NO `{{...}}` placeholders remain anywhere:
   ```python
   import zipfile
   with zipfile.ZipFile('/root/clinic_intake_ready.hwpx', 'r') as z:
       for name in z.namelist():
           try:
               content = z.read(name).decode('utf-8', errors='ignore')
               matches = re.findall(r'\{\{.*?\}\}', content)
               if matches:
                   print(f'FAIL: {name} still has placeholders: {matches}')
           except:
               pass
   print('Validation complete')
   ```
5. Verify the age note is present with correct format `(<N>세)`
6. Verify the phone number is in `NNN-NNNN-NNNN` format
7. Verify Korean labels and handwritten-signature note are preserved
8. Check that modified paragraphs do not contain layout-cache elements by inspecting the XML of a few modified paragraphs

## Critical reminders
- Every single `{{...}}` must be replaced - search ALL files in the ZIP, not just the obvious section files
- The patient name likely appears multiple times (including a confirmation line) - replace ALL occurrences
- The age note `(<N>세)` must be ADDED after the birth date, not replace it
- Keep all existing Korean text labels intact
- The output path must be exactly `/root/clinic_intake_ready.hwpx`

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