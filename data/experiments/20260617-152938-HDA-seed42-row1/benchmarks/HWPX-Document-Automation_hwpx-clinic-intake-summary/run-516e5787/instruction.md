# Task Instruction

Complete the clinic intake summary by filling the HWPX template with patient data. Follow these steps precisely:

## Step 1: Inspect the workspace
```
ls -la /root/
cat /root/patient_intake.json
```
Understand the JSON structure and all available field values.

## Step 2: Examine the HWPX template structure
```
cd /root
mkdir -p /tmp/hwpx_inspect
cp clinic_intake_template.hwpx /tmp/hwpx_inspect/
cd /tmp/hwpx_inspect
python3 -c "import zipfile; z=zipfile.ZipFile('clinic_intake_template.hwpx','r'); print('\n'.join(z.namelist()))"
```
List all files in the HWPX ZIP package.

## Step 3: Extract and inspect all XML content files
Extract the HWPX package and examine every XML file for `{{` placeholder patterns. HWPX content is typically in files like `Contents/section0.xml` or similar. Read each XML file fully:
```
python3 -c "
import zipfile
z = zipfile.ZipFile('clinic_intake_template.hwpx', 'r')
for name in z.namelist():
    try:
        content = z.read(name).decode('utf-8')
        if '{{' in content or '}}' in content:
            print(f'=== {name} ===')
            print(content)
    except:
        pass
"
```
Also print all XML files to understand the full structure, especially to identify layout-cache elements.

## Step 4: CRITICAL - Check for split placeholders
Placeholders like `{{patient_name}}` may be split across multiple XML text runs (e.g., `{{patient`, `_name}}`). You MUST handle this. Search for `{{` and `}}` patterns and check if they appear in the same text node or are split across sibling elements.

## Step 5: Write a Python script to perform all replacements
Create `/tmp/fill_template.py` that does the following:

### 5a: Load patient_intake.json
Read all values from the JSON file.

### 5b: Calculate Korean full-year age (만 나이)
Compute the age as: the number of full years between the birth date and the visit date. This is standard Western/만 age: if the birthday has not yet occurred in the visit year, subtract 1 from (visit_year - birth_year). Format as `(<N>세)` — e.g., `(45세)`.

### 5c: Normalize phone number
Strip all non-digit characters from the callback phone number, then format as `XXX-XXXX-XXXX` (3-4-4 grouping, i.e., `000-0000-0000`).

### 5d: Process the HWPX package
- Open the template HWPX as a ZIP.
- For each file in the ZIP, read its content.
- For XML files that contain placeholders:
  - Parse the XML properly using `lxml.etree` or `xml.etree.ElementTree`.
  - CRITICAL: Handle split placeholders. Concatenate text across adjacent text runs within the same paragraph, find placeholder boundaries, perform replacement, and redistribute text back. One reliable approach: for each paragraph element, collect all text-bearing child elements in order, join their text, perform all replacements on the joined string, then assign the full replaced text to the first text element and clear the rest (or delete the extra elements).
  - After replacing text in a paragraph, remove any layout-cache child elements from that paragraph. These are typically elements with tag names containing `linesegarray`, `lineSegArray`, `LineSeg`, or similar layout/cache namespace elements. Inspect the actual XML to identify the correct element names. Common HWPX layout cache elements include elements in namespaces related to `hp:linesegarray` or `hp:lineSegArray` or child elements like `<hp:linesegarray>`. Remove them from any paragraph where text was modified.
  - Also check for and remove any `<hp:charPrIDRef>` or similar cached positioning if present in modified runs.
  - After the birth date value, append ` (<N>세)` with the calculated age.
  - Replace ALL occurrences of each placeholder, including repeated ones (e.g., patient name confirmation line).
- Write all files (modified and unmodified) to a new ZIP at `/root/clinic_intake_ready.hwpx`, preserving the original ZIP structure and compression.

### 5e: Build the placeholder mapping
Map each `{{placeholder}}` to its replacement value from the JSON. Common mappings might include patient name, birth date, visit date, phone number, address, symptoms, etc. Print all placeholders found in the template so you can verify the mapping is complete.

## Step 6: Run the script
```
python3 /tmp/fill_template.py
```

## Step 7: Validate the output
```
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/clinic_intake_ready.hwpx', 'r')
print('Valid ZIP: OK')
print('Files:', z.namelist())
for name in z.namelist():
    try:
        content = z.read(name).decode('utf-8')
        if '{{' in content:
            print(f'ERROR: Remaining placeholder in {name}')
            # Print context around the placeholder
            idx = content.find('{{')
            print(content[max(0,idx-50):idx+80])
        if '}}' in content:
            # Check it's not part of valid XML
            import re
            matches = re.findall(r'\{\{.*?\}\}', content)
            if matches:
                print(f'ERROR: Remaining placeholders in {name}: {matches}')
    except:
        pass
print('Validation complete')
"
```

Also verify:
- The age note appears after the birth date in format `(<N>세)`
- The phone number is in `000-0000-0000` format
- Korean labels are preserved
- No `{{...}}` remains anywhere
- The file is a valid ZIP/HWPX package

## Step 8: If any issues found, fix and re-validate
Re-read the problematic XML, fix the script, regenerate, and re-validate until all requirements pass.

## IMPORTANT NOTES:
- Do NOT use simple string replacement on raw XML — this can break XML structure. Parse XML properly.
- Handle the case where placeholder text spans multiple XML text runs.
- The age format must be exactly `(<N>세)` with parentheses, appended after the birth date with a space.
- Layout cache elements MUST be removed from any paragraph whose text content was modified. Inspect the actual XML element names/namespaces used in the template to identify them correctly.
- Preserve all other document structure, styles, and content exactly.

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