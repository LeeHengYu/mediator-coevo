# Task Instruction

Execute the following steps to fill in the training feedback HWPX template and produce `/root/training_feedback_ready.hwpx`.

## Step 1 — Inspect the workspace

```bash
ls /root/
find /root/ -name 'training_feedback*' -type f 2>/dev/null
```

Identify the exact paths of `training_feedback_template.hwpx` and `training_feedback.json`.

## Step 2 — Read the JSON data

```bash
cat /root/training_feedback.json 2>/dev/null || find / -name 'training_feedback.json' -type f 2>/dev/null
```

Note every key-value pair. Pay special attention to `참석자수`, `만족도`, and `종합의견`.

## Step 3 — Explore the HWPX package structure

```bash
cd /root
mkdir -p /tmp/hwpx_work
cp training_feedback_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip -o template.zip -d template_extracted
find template_extracted -type f | sort
```

List every file inside the ZIP. Identify all `section*.xml` files (typically under `Contents/`) and any other XML files that could contain `{{...}}` placeholders.

## Step 4 — Find all placeholders

```bash
grep -rn '{{' template_extracted/
```

Record every placeholder and which file it appears in. Confirm each placeholder has a matching JSON key.

## Step 5 — Write and run the Python automation script

Create `/tmp/hwpx_work/fill.py` with the following logic:

```python
import json, os, re, zipfile, shutil

# --- paths ---
TEMPLATE = '/root/training_feedback_template.hwpx'  # adjust if different
JSON_FILE = '/root/training_feedback.json'           # adjust if different
OUTPUT   = '/root/training_feedback_ready.hwpx'
WORK     = '/tmp/hwpx_fill'

if os.path.exists(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK)

# --- load JSON ---
with open(JSON_FILE, encoding='utf-8') as f:
    data = json.load(f)

# --- build replacement map ---
replacements = {}
for k, v in data.items():
    v_str = str(v)
    if k == '참석자수':
        # digits only
        v_str = re.sub(r'[^0-9]', '', v_str)
    elif k == '만족도':
        # extract numeric score, format as "X.X점 (5.0점 만점)"
        score = re.search(r'[\d.]+', v_str)
        if score:
            v_str = f'{score.group()}점 (5.0점 만점)'
    elif k == '종합의견':
        # append the required sentence
        v_str = v_str.rstrip()
        if not v_str.endswith('.'):
            v_str += '.'
        v_str = v_str + ' 후속 심화반 검토 요망.'
    replacements['{{' + k + '}}'] = v_str

print('Replacement map:')
for k, v in replacements.items():
    print(f'  {k} -> {v}')

# --- extract template ---
with zipfile.ZipFile(TEMPLATE, 'r') as zin:
    entry_list = zin.namelist()
    zin.extractall(WORK)

print('\nEntries:', entry_list)

# --- process every file, replace placeholders, remove layout cache ---
import xml.etree.ElementTree as ET

for entry in entry_list:
    fpath = os.path.join(WORK, entry)
    if os.path.isdir(fpath):
        continue
    # Only process XML files for placeholder replacement
    if not entry.endswith('.xml'):
        continue
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        continue

    if '{{' not in content:
        continue

    print(f'\nProcessing: {entry}')
    modified = False

    # Do replacements
    for placeholder, value in replacements.items():
        if placeholder in content:
            content = content.replace(placeholder, value)
            modified = True
            print(f'  Replaced {placeholder}')

    if modified:
        # Remove lineSegArray elements (layout cache) from modified paragraphs
        # We'll use regex to remove all <hp:lineSegArray>...</hp:lineSegArray>
        # and also <lineSegArray>...</lineSegArray> (namespace variants)
        content_before = content
        content = re.sub(r'<[^>]*:?lineSegArray[^>]*>.*?</[^>]*:?lineSegArray>', '', content, flags=re.DOTALL)
        # Also handle self-closing
        content = re.sub(r'<[^>]*:?lineSegArray[^/]*/>', '', content, flags=re.DOTALL)
        if content != content_before:
            print('  Removed lineSegArray elements')

        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)

# --- check no placeholders remain ---
remaining = []
for entry in entry_list:
    fpath = os.path.join(WORK, entry)
    if os.path.isdir(fpath):
        continue
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read()
        found = re.findall(r'\{\{[^}]+\}\}', text)
        if found:
            remaining.extend([(entry, m) for m in found])
    except Exception:
        pass

if remaining:
    print('\n*** WARNING: remaining placeholders ***')
    for entry, m in remaining:
        print(f'  {entry}: {m}')
else:
    print('\nNo remaining placeholders — good.')

# --- repackage as HWPX (ZIP) ---
# mimetype must be first entry, stored uncompressed
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)

with zipfile.ZipFile(OUTPUT, 'w') as zout:
    for entry in entry_list:
        fpath = os.path.join(WORK, entry)
        if os.path.isdir(fpath):
            continue
        if entry == 'mimetype':
            zout.write(fpath, entry, compress_type=zipfile.ZIP_STORED)
        else:
            zout.write(fpath, entry, compress_type=zipfile.ZIP_DEFLATED)

print(f'\nOutput written to {OUTPUT}')
print('Output size:', os.path.getsize(OUTPUT))
```

Run the script:
```bash
python3 /tmp/hwpx_work/fill.py
```

## Step 6 — Validate the output

```bash
# Confirm it's a valid ZIP
python3 -c "import zipfile; z=zipfile.ZipFile('/root/training_feedback_ready.hwpx','r'); print(z.namelist()); z.close()"

# Confirm no placeholders remain in any XML
python3 -c "
import zipfile, re
z = zipfile.ZipFile('/root/training_feedback_ready.hwpx','r')
for name in z.namelist():
    try:
        text = z.read(name).decode('utf-8')
        found = re.findall(r'\{\{[^}]+\}\}', text)
        if found: print(f'PLACEHOLDER in {name}: {found}')
    except: pass
print('Placeholder check done.')
z.close()
"

# Confirm 참석자수 is digits only
python3 -c "
import zipfile, re
z = zipfile.ZipFile('/root/training_feedback_ready.hwpx','r')
for name in z.namelist():
    try:
        text = z.read(name).decode('utf-8')
        if '참석자수' in text or '명' in text:
            # look for the attendance value context
            for line in text.split('\n'):
                if '참석' in line or '명' in line:
                    print(f'{name}: {line.strip()[:200]}')
    except: pass
z.close()
"

# Confirm 만족도 formatting
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/training_feedback_ready.hwpx','r')
for name in z.namelist():
    try:
        text = z.read(name).decode('utf-8')
        if '만점' in text:
            for line in text.split('\n'):
                if '만점' in line:
                    print(f'{name}: {line.strip()[:200]}')
    except: pass
z.close()
"

# Confirm 종합의견 has appended sentence
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/training_feedback_ready.hwpx','r')
for name in z.namelist():
    try:
        text = z.read(name).decode('utf-8')
        if '후속 심화반' in text:
            for line in text.split('\n'):
                if '후속' in line:
                    print(f'{name}: {line.strip()[:200]}')
    except: pass
z.close()
"

# Confirm no lineSegArray in modified sections
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/training_feedback_ready.hwpx','r')
for name in z.namelist():
    try:
        text = z.read(name).decode('utf-8')
        if 'lineSegArray' in text:
            print(f'WARNING: lineSegArray found in {name}')
    except: pass
print('lineSegArray check done.')
z.close()
"
```

## Step 7 — Run the verifier (if present)

```bash
cd /root && ls test_output.py 2>/dev/null && python3 -m pytest test_output.py -v
```

If the test suite exists, run it and confirm all tests pass. If any test fails, read the failure message, fix the issue, and re-run.

## Important Notes

- **Namespace handling**: The placeholder `{{key}}` may be split across XML tags (e.g., `<hp:t>{{</hp:t><hp:t>key}}</hp:t>`). After the initial grep in Step 4, if you see placeholders split across tags, adjust the script to work on the raw XML text (string replacement) rather than parsed DOM, which the current approach already does.
- **mimetype entry**: Must be the FIRST entry in the ZIP and stored uncompressed (ZIP_STORED). The script preserves the original entry order from the template.
- **lineSegArray removal**: Critical for preventing overlapping characters when the HWPX renderer opens the file. Remove from ALL section XMLs that were modified.
- **종합의견 period handling**: The JSON value may or may not end with a period. The script normalizes this before appending `후속 심화반 검토 요망.`
- **참석자수**: Extract digits only (e.g., "32명" → "32", "32" → "32").
- **만족도**: Extract the numeric score and format as `X.X점 (5.0점 만점)` (e.g., "4.5" → "4.5점 (5.0점 만점)", "4.5/5.0" → "4.5점 (5.0점 만점)").
- If file paths differ from assumed `/root/`, adjust accordingly based on Step 1 findings.

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