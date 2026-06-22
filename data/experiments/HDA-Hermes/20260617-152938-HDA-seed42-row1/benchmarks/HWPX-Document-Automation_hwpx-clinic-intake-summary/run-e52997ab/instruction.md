# Task Instruction

Complete the clinic intake summary by filling the HWPX template with patient data. Follow these steps exactly:

## Step 1: Inspect the workspace
```bash
ls /root/
find /root/ -name '*.hwpx' -o -name '*.json' | head -20
```

## Step 2: Read the patient data
```bash
cat /root/patient_intake.json
```
Note all field names and values. Pay special attention to:
- Patient name (may appear multiple times in template)
- Birth date and visit date (needed for age calculation)
- Phone number (needs normalization to 000-0000-0000)

## Step 3: Examine the HWPX template structure
The HWPX file is a ZIP archive containing XML files. List its contents:
```bash
python3 -c "import zipfile; z=zipfile.ZipFile('/root/clinic_intake_template.hwpx'); print('\n'.join(z.namelist()))"
```
Then inspect the main content XML (likely `Contents/section0.xml` or similar) to find all `{{...}}` placeholders.

## Step 4: Write and run the Python script
Create `/root/solve.py` that does the following:

1. **Load patient_intake.json** and extract all values.

2. **Normalize the phone number**: Strip all non-digit characters, then format as `000-0000-0000` (3-4-4 grouping).

3. **Calculate Korean full-year age (만 나이)**: Using the birth date and visit date from the JSON. Age = visit_year - birth_year, minus 1 if the visit date hasn't reached the birthday yet in that year. Format as `(<N>세)`.

4. **Open the HWPX zip**, iterate over all entries. For each XML entry (especially section XML files):
   a. Parse with `xml.etree.ElementTree`.
   b. Register all namespaces found in the file to preserve them on output.
   c. For each `<hp:p>` paragraph element, concatenate all text content from child `<hp:t>` (or similar text run) elements to form the full paragraph text. This is critical because placeholders like `{{patient_name}}` may be split across multiple text run elements.
   d. If the concatenated text contains any `{{...}}` placeholder, perform all replacements on the concatenated text.
   e. For the birth date placeholder, after replacing it with the actual birth date value, append the age note ` (<N>세)` immediately after.
   f. After replacement, redistribute the replaced text back into the text run elements (simplest: put all text into the first `<hp:t>` element and clear the rest).
   g. For any paragraph that was modified, remove all `<hp:lineSegArray>` child elements (layout cache) so the document renders cleanly.

5. **Build the output HWPX**: Create a new zip at `/root/clinic_intake_ready.hwpx`. Copy all entries from the original, replacing modified XML entries with the updated versions. Preserve compression type.

6. **Validate**: After writing, re-open the output zip and scan ALL XML content for any remaining `{{` to ensure no placeholders survive.

Key implementation details:
- Use `xml.etree.ElementTree` with namespace registration.
- The HWPX namespace for paragraphs is typically `http://www.hancom.co.kr/hwpml/2011/paragraph` with prefix `hp`.
- Text nodes may be in elements like `{http://www.hancom.co.kr/hwpml/2011/paragraph}t` nested inside run elements.
- When scanning for placeholders, search the concatenated text of all text nodes within a paragraph, not individual nodes.
- Make sure to handle ALL XML files in the archive, not just section0.xml.

Run the script:
```bash
python3 /root/solve.py
```

## Step 5: Verify the output
```bash
# Check file exists and is a valid zip
python3 -c "import zipfile; z=zipfile.ZipFile('/root/clinic_intake_ready.hwpx'); print('Valid ZIP, entries:', len(z.namelist()))"

# Check no placeholders remain
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/clinic_intake_ready.hwpx')
for name in z.namelist():
    try:
        content = z.read(name).decode('utf-8', errors='ignore')
        if '{{' in content:
            print(f'PLACEHOLDER FOUND in {name}:', [s for s in content.split('{{')[1:]])
    except: pass
print('Placeholder check complete')
"

# Check age note is present
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/clinic_intake_ready.hwpx')
for name in z.namelist():
    content = z.read(name).decode('utf-8', errors='ignore')
    if '세)' in content:
        print(f'Age note found in {name}')
"

# Check phone format
python3 -c "
import zipfile, re
z = zipfile.ZipFile('/root/clinic_intake_ready.hwpx')
for name in z.namelist():
    content = z.read(name).decode('utf-8', errors='ignore')
    phones = re.findall(r'\d{3}-\d{4}-\d{4}', content)
    if phones: print(f'Phone in {name}:', phones)
"

# Check no lineSegArray in modified content
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/clinic_intake_ready.hwpx')
for name in z.namelist():
    content = z.read(name).decode('utf-8', errors='ignore')
    if 'lineSegArray' in content:
        print(f'WARNING: lineSegArray still in {name}')
"
```

If the test suite exists, also run:
```bash
cd /root && python3 -m pytest test_output.py -v
```

Ensure all checks pass before finishing.

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