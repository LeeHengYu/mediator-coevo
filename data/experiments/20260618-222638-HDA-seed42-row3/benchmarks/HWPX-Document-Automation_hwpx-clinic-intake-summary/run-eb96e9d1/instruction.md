# Task Instruction

Execute the following steps carefully to produce `/root/clinic_intake_ready.hwpx`.

## 1  Explore the workspace
```bash
ls /root/
find /root/ -name 'patient_intake.json' -o -name 'clinic_intake_template.hwpx' 2>/dev/null
```
Identify the exact paths of the template and JSON file.

## 2  Inspect the JSON data
```bash
cat /root/patient_intake.json  # or wherever it is found
```
Note every key-value pair. Pay special attention to: patient name, birth date, visit date, phone number, and any other fields.

## 3  Inspect the HWPX template structure
An `.hwpx` file is a ZIP archive. Unzip and inspect:
```bash
mkdir -p /tmp/hwpx_work
cp <template_path> /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
unzip -o template.hwpx -d template_contents
find template_contents -type f | sort
```
Then read every XML file that could contain text content:
```bash
for f in $(find template_contents -name '*.xml'); do echo "===== $f ====="; cat "$f"; done
```

## 4  Identify ALL placeholders
Search for every `{{` occurrence across ALL files in the unzipped archive:
```bash
grep -rn '{{' template_contents/
```
Also search for partial placeholder fragments that may be split across multiple `<hp:t>` tags:
```bash
grep -rn '{' template_contents/ | grep -v 'xmlns' | grep -v 'http'
```
Record every placeholder and which file it appears in. Pay close attention to whether any placeholder is split across multiple XML elements (e.g., `<hp:t>{{birth</hp:t>...</hp:t>_date}}</hp:t>`).

## 5  Write a Python script to perform all replacements
Create `/tmp/hwpx_work/fill_template.py` with the following logic:

```python
import json, os, re, shutil, zipfile
from pathlib import Path

# --- Paths ---
TEMPLATE = '<absolute path to template.hwpx>'
JSON_FILE = '<absolute path to patient_intake.json>'
OUTPUT = '/root/clinic_intake_ready.hwpx'
WORK = '/tmp/hwpx_fill'

# --- Load data ---
with open(JSON_FILE) as f:
    data = json.load(f)

# --- Compute derived values ---
# Korean full-year age: visit_year - birth_year
from datetime import date
birth_date_str = data['birth_date']       # e.g. '1990-11-02'
visit_date_str = data['visit_date']       # e.g. '2025-06-15'
birth_year = int(birth_date_str[:4])
visit_year = int(visit_date_str[:4])
age = visit_year - birth_year
# The age note to append: ' (35세)'
age_note = f' ({age}세)'

# --- Normalize phone: digits only -> 000-0000-0000 ---
raw_phone = data.get('phone') or data.get('callback_phone') or data.get('phone_number') or ''
digits = re.sub(r'\D', '', raw_phone)
if len(digits) == 11:
    phone_formatted = f'{digits[:3]}-{digits[3:7]}-{digits[7:]}'
elif len(digits) == 10:
    phone_formatted = f'{digits[:3]}-{digits[3:6]}-{digits[6:]}'
else:
    phone_formatted = raw_phone  # fallback

# --- Build replacement map: placeholder -> value ---
# Map every JSON key to its placeholder form {{key}}
replace_map = {}
for k, v in data.items():
    replace_map['{{' + k + '}}'] = str(v)

# Override specific ones with computed values:
# birth_date placeholder gets date + age note
birth_key = None
for k in data:
    if 'birth' in k.lower():
        birth_key = k
        break
if birth_key:
    replace_map['{{' + birth_key + '}}'] = data[birth_key] + age_note

# phone placeholder gets formatted phone
for k in data:
    if 'phone' in k.lower() or 'callback' in k.lower():
        replace_map['{{' + k + '}}'] = phone_formatted

print('Replacement map:', replace_map)

# --- Unzip ---
if os.path.exists(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK)
with zipfile.ZipFile(TEMPLATE) as zf:
    zf.extractall(WORK)

# --- Process every XML file ---
for xml_path in Path(WORK).rglob('*.xml'):
    with open(xml_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # CRITICAL: Handle placeholders split across multiple <hp:t> tags.
    # Strategy: First, merge adjacent <hp:t> elements within the same <hp:run>,
    # then do simple string replacement.
    # Merge: collapse </hp:t></hp:t> boundaries (possibly with </hp:t><hp:t> patterns)
    # More robust: extract all text, find placeholders, rebuild.
    
    # Step A: Merge split placeholders by removing tag boundaries inside {{ }}
    # Find sequences like: {{...split across tags...}}
    # Use regex to find and merge:
    # Pattern: remove </hp:t> and <hp:t> (and any attributes) between {{ and }}
    
    MAX_ITER = 20
    for _ in range(MAX_ITER):
        # Remove XML tags that appear between { and } characters
        new_content = re.sub(
            r'(\{\{[^}]*?)</hp:t>\s*(?:<[^>]*>)*\s*<hp:t[^>]*>([^}]*?\}\})',
            r'\1\2',
            content
        )
        # Also handle cases where the opening {{ is split
        new_content = re.sub(
            r'(\{)</hp:t>\s*(?:<[^>]*>)*\s*<hp:t[^>]*>(\{[a-zA-Z_])',
            r'\1\2',
            new_content
        )
        # Handle closing }} split
        new_content = re.sub(
            r'(\})</hp:t>\s*(?:<[^>]*>)*\s*<hp:t[^>]*>(\})',
            r'\1\2',
            new_content
        )
        if new_content == content:
            break
        content = new_content

    # Step B: Simple string replacement for all placeholders
    for placeholder, value in replace_map.items():
        content = content.replace(placeholder, value)

    # Step C: Remove stale layout-cache elements (hp:linesegarray, hp:lineSegArray, etc.)
    # These cause overlapping characters when text length changes.
    content = re.sub(r'<hp:linesegarray[^>]*>.*?</hp:linesegarray>', '', content, flags=re.DOTALL|re.IGNORECASE)
    content = re.sub(r'<hp:lineSegArray[^>]*>.*?</hp:lineSegArray>', '', content, flags=re.DOTALL)
    content = re.sub(r'<hp:lineseg[^/]*?/>', '', content, flags=re.IGNORECASE)
    # Also remove layoutcache or similar elements
    content = re.sub(r'<hp:layoutcache[^>]*>.*?</hp:layoutcache>', '', content, flags=re.DOTALL|re.IGNORECASE)
    
    if content != original:
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Modified: {xml_path}')

# --- Verify no remaining placeholders ---
remaining = []
for xml_path in Path(WORK).rglob('*.xml'):
    with open(xml_path, 'r', encoding='utf-8') as f:
        text = f.read()
    found = re.findall(r'\{\{.*?\}\}', text)
    if found:
        remaining.append((str(xml_path), found))
        # Also check for split placeholders
    # Check for orphan { that might indicate split placeholders
    orphans = re.findall(r'\{\{[^}]+$|^[^{]*\}\}', text, re.MULTILINE)
    if orphans:
        remaining.append((str(xml_path), 'ORPHANS: ' + str(orphans)))

if remaining:
    print('WARNING: Remaining placeholders:', remaining)
    # Attempt more aggressive merging or manual fix
else:
    print('All placeholders replaced successfully.')

# --- Repackage as .hwpx (ZIP) ---
# Preserve the original ZIP structure
with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zout:
    for fpath in sorted(Path(WORK).rglob('*')):
        if fpath.is_file():
            arcname = str(fpath.relative_to(WORK))
            zout.write(fpath, arcname)

print(f'Output written to {OUTPUT}')

# --- Final verification: check the output ---
with zipfile.ZipFile(OUTPUT) as zf:
    for name in zf.namelist():
        if name.endswith('.xml'):
            with zf.open(name) as xf:
                xtext = xf.read().decode('utf-8')
                if '{{' in xtext:
                    print(f'ERROR: Remaining placeholder in {name}')
                if birth_date_str + age_note in xtext:
                    print(f'FOUND birth date + age in {name}: {birth_date_str}{age_note}')
```

**IMPORTANT ADJUSTMENTS TO MAKE IN THE SCRIPT:**
- After reading the JSON, print all keys to discover the exact key names (they might be `birth_date`, `birthdate`, `date_of_birth`, etc.).
- After grepping for placeholders, adapt the replacement map to match the exact placeholder names found.
- The phone key might be named differently — check the JSON keys.
- Korean full-year age (만 나이 or 한국 나이): The task says "Korean full-year age" which is `visit_year - birth_year`. This gives 35 for birth year 1990 and visit year 2025.

## 6  Run the script
```bash
python3 /tmp/hwpx_work/fill_template.py
```

## 7  Verify the output thoroughly
```bash
# Check it's a valid ZIP
unzip -t /root/clinic_intake_ready.hwpx

# Extract and check section0.xml for the birth date + age
mkdir -p /tmp/verify
cd /tmp/verify
unzip -o /root/clinic_intake_ready.hwpx -d verify_contents

# Search for the expected birth date + age string
grep -r '1990-11-02' verify_contents/ || echo 'Birth date NOT FOUND'
grep -r '세)' verify_contents/ || echo 'Age note NOT FOUND'

# Verify NO remaining placeholders
grep -rn '{{' verify_contents/ && echo 'ERROR: Placeholders remain!' || echo 'OK: No placeholders remain'

# Check patient name appears (including repeated confirmation line)
grep -rn 'patient_name_value_here' verify_contents/  # replace with actual name from JSON

# Print the full section0.xml to visually inspect
cat verify_contents/Contents/section0.xml
```

## 8  If verification fails
- Re-read the XML to understand the exact tag structure around the failing placeholder.
- If placeholders are split across tags, manually merge the `<hp:t>` elements in the source XML before replacement.
- Re-run and re-verify.

## Key points from previous failure
The previous attempt failed because `1990-11-02 (35세)` was not found in section0.xml. This likely means:
1. The birth date placeholder was split across XML tags and the replacement didn't work, OR
2. The age note wasn't appended correctly.

You MUST verify that the exact string `1990-11-02 (35세)` (with a space before the opening parenthesis) appears in the final section0.xml. If it doesn't, debug by printing the XML around where the birth date placeholder was and fix the merging/replacement logic.

Also ensure the `<hp:linesegarray>` or similar layout cache elements are removed from any paragraph whose text was modified, as specified in the requirements.

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