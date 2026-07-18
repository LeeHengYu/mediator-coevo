# Task Instruction

You must produce a Python script and run it to fill in the clinic intake template and save the result.

## Step 0 – Inspect available files
```bash
ls /root/
find /root/ -name '*.hwpx' -o -name '*.json' 2>/dev/null
```
Read the patient data:
```bash
cat /root/patient_intake.json
```
Examine the HWPX template structure:
```bash
python3 -c "
import zipfile, sys
with zipfile.ZipFile('/root/clinic_intake_template.hwpx') as z:
    for name in z.namelist():
        print(name)
"
```
Then dump the main content XML (likely `Contents/section0.xml`) to see all placeholders:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('/root/clinic_intake_template.hwpx') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            print(f'=== {name} ===')
            print(z.read(name).decode('utf-8'))
"
```
Carefully note every `{{...}}` placeholder and their exact names.

## Step 1 – Build and run the generation script

Create `/root/generate.py` with the following logic:

1. **Load `patient_intake.json`** – parse all fields.
2. **Compute Korean full-year age**: from the patient's birth date to the visit date. Korean full-year age = `visit_year - birth_year`, adjusted down by 1 if the visit date is before the birthday in the visit year. Format as `(<N>세)`.
3. **Normalize phone number**: strip all non-digit characters, then format as `NNN-NNNN-NNNN` (3-4-4 grouping). If the number starts with a leading zero and has 11 digits this is the standard Korean mobile format.
4. **Open the HWPX ZIP**, iterate over every entry. For each `.xml` file:
   a. Read the XML content as a UTF-8 string.
   b. Perform placeholder replacement using Python `str.replace()` for each `{{placeholder}}` → value mapping. Make sure to handle the birth-date placeholder specially: replace `{{birth_date}}` (or whatever the exact placeholder name is) with the birth date value followed by a space and the age note, e.g., `1990-05-15 (33세)`. **Inspect the actual placeholder names from Step 0 and map them precisely.**
   c. Handle repeated placeholders (e.g., patient name may appear multiple times).
   d. After all replacements, verify no `{{` or `}}` remain in the text. If any do, print a warning with the remaining placeholder text so you can fix the mapping.
   e. **Remove stale layout-cache elements from modified paragraphs**: Parse the replaced XML with `lxml.etree` (or `xml.etree.ElementTree`). For every `<hp:p>` element (or equivalent paragraph tag – check the actual namespace), remove child elements that are layout caches such as `<hp:lineSegArray>`, `<hp:lineseg>`, `<hp:lineSegArray>` and similar. The cross-task artifact confirms this is critical for clean rendering.
   f. Serialize the XML back to a UTF-8 string.
5. **Write the output HWPX** to `/root/clinic_intake_ready.hwpx` as a new ZIP, copying all non-XML entries verbatim and writing the modified XML entries.
6. **Preserve Korean labels and the handwritten-signature note** – do NOT remove or alter any text that is not a placeholder.

## Step 2 – Validate
After running the script:
```bash
python3 /root/generate.py
```
Then validate:
```bash
# Check it's a valid ZIP
python3 -c "
import zipfile
with zipfile.ZipFile('/root/clinic_intake_ready.hwpx') as z:
    z.testzip()
    print('Valid ZIP')
"

# Check no placeholders remain
python3 -c "
import zipfile, re
with zipfile.ZipFile('/root/clinic_intake_ready.hwpx') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            matches = re.findall(r'\{\{.*?\}\}', content)
            if matches:
                print(f'REMAINING PLACEHOLDERS in {name}: {matches}')
            else:
                print(f'{name}: OK, no placeholders')
"

# Dump final XML to verify age note, phone format, content
python3 -c "
import zipfile
with zipfile.ZipFile('/root/clinic_intake_ready.hwpx') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            print(f'=== {name} ===')
            print(z.read(name).decode('utf-8')[:5000])
"
```

Check specifically:
- Age note like `(33세)` appears right after the birth date.
- Phone is in `000-0000-0000` format (digits and hyphens only).
- Patient name appears in all locations (including confirmation line).
- Korean labels and signature note are intact.
- No `<hp:lineSegArray>` or `<hp:lineseg>` elements remain inside any paragraph whose text was modified.
- No `{{...}}` text anywhere.

If anything is wrong, fix the script and re-run. Do not consider the task complete until all checks pass.

## Step 3 – Run the verifier if available
```bash
ls /root/test_output.py 2>/dev/null && python3 -m pytest /root/test_output.py -v
```
If there is a test file, run it and ensure all tests pass. Fix any failures before finishing.

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