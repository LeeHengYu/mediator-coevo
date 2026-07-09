# Task Instruction

## Task: Fill in training feedback HWPX template

### Goal
Replace all `{{...}}` placeholders in `training_feedback_template.hwpx` with values from `training_feedback.json`, apply specific formatting rules, and save the result to `/root/training_feedback_ready.hwpx`.

### Step-by-step plan

#### 1. Understand the HWPX format
- A `.hwpx` file is a ZIP-based package (like DOCX/ODP) containing XML files.
- `cd /root` and list available files. Locate `training_feedback_template.hwpx` and `training_feedback.json`.
- Run: `file training_feedback_template.hwpx` to confirm it's a ZIP.
- Run: `python3 -c "import zipfile; z=zipfile.ZipFile('training_feedback_template.hwpx'); print(z.namelist())"` to list all entries.

#### 2. Inspect the JSON data
- `cat training_feedback.json` — read all key-value pairs. Note every key name exactly.

#### 3. Inspect the template XML files
- Extract the HWPX to a temp directory: `mkdir /tmp/hwpx_work && cd /tmp/hwpx_work && unzip /root/training_feedback_template.hwpx`
- Search for `{{` across ALL extracted files: `grep -rl '{{' /tmp/hwpx_work/`
- For each file containing placeholders, `cat` the full file and note every `{{...}}` token and its surrounding XML context.
- Also look for layout-cache elements. Common HWPX cache element names include `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:lineseg>`, `<hp:lineSeg>`, `<hp:paraPr>` sub-elements related to caching, or `<hp:charPr>` width caches. Identify the exact element names present.

#### 4. Write a Python script to perform all replacements
Create `/root/fill_template.py` that does the following:

```python
import json, zipfile, os, re, shutil, copy
from pathlib import Path

# Load JSON
with open('/root/training_feedback.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Prepare transformed values:
# a) 참석자수: extract digits only (e.g., "35명" -> "35")
if '참석자수' in data:
    data['참석자수'] = re.sub(r'[^0-9]', '', str(data['참석자수']))

# b) 만족도: rewrite as "X.X점 (5.0점 만점)" preserving the numeric score
if '만족도' in data:
    score = data['만족도']
    # Extract numeric value if it contains extra text
    m = re.search(r'([\d.]+)', str(score))
    if m:
        numeric = m.group(1)
        data['만족도'] = f"{numeric}점 (5.0점 만점)"

# c) 종합의견 (or whatever the overall-opinion key is): append sentence
# Identify the key — look for '종합의견' or similar
for key in list(data.keys()):
    if '종합' in key or '의견' in key:
        data[key] = str(data[key]).rstrip() + ' 후속 심화반 검토 요망.'
        break

# Work directory
work = Path('/tmp/hwpx_edit')
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)

# Extract
with zipfile.ZipFile('/root/training_feedback_template.hwpx', 'r') as zin:
    zin.extractall(work)

# Find and replace in all files
for fpath in work.rglob('*'):
    if fpath.is_file():
        try:
            content = fpath.read_text(encoding='utf-8')
        except (UnicodeDecodeError, PermissionError):
            continue
        if '{{' not in content:
            continue
        
        original = content
        # Replace each {{key}} with its value
        for key, val in data.items():
            content = content.replace('{{' + key + '}}', str(val))
        
        # Also handle possible {{key}} patterns with spaces like {{ key }}
        for key, val in data.items():
            content = re.sub(r'\{\{\s*' + re.escape(key) + r'\s*\}\}', str(val), content)
        
        if content != original:
            # Remove stale layout-cache elements from modified paragraphs
            # Remove <hp:linesegarray>...</hp:linesegarray> (case-insensitive tag variations)
            content = re.sub(
                r'<[a-zA-Z:]*[Ll]ine[Ss]eg[Aa]rray[^>]*>.*?</[a-zA-Z:]*[Ll]ine[Ss]eg[Aa]rray>',
                '', content, flags=re.DOTALL
            )
            
            fpath.write_text(content, encoding='utf-8')

# Verify no {{...}} remain
remaining = []
for fpath in work.rglob('*'):
    if fpath.is_file():
        try:
            content = fpath.read_text(encoding='utf-8')
        except:
            continue
        found = re.findall(r'\{\{[^}]+\}\}', content)
        if found:
            remaining.append((str(fpath), found))

if remaining:
    print('WARNING: Unreplaced placeholders found:')
    for fp, tokens in remaining:
        print(f'  {fp}: {tokens}')
else:
    print('All placeholders replaced successfully.')

# Repackage as HWPX (ZIP)
output = '/root/training_feedback_ready.hwpx'
with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zout:
    for fpath in sorted(work.rglob('*')):
        if fpath.is_file():
            arcname = str(fpath.relative_to(work))
            zout.write(fpath, arcname)

print(f'Output written to {output}')

# Final check
with zipfile.ZipFile(output, 'r') as z:
    print(f'ZIP entries: {len(z.namelist())}')
    # Verify no placeholders in output
    for name in z.namelist():
        try:
            txt = z.read(name).decode('utf-8')
        except:
            continue
        if '{{' in txt:
            print(f'  REMAINING PLACEHOLDER in {name}')
```

**IMPORTANT**: Before writing this script, you MUST first complete steps 1-3 to inspect the actual file structure, JSON keys, placeholder names, and XML element names. Then adapt the script accordingly. The script above is a starting template — you must adjust:
- The exact JSON keys (especially the overall-opinion key)
- The layout-cache element removal regex to match the actual element names found in step 3
- Handle any case where placeholders might be split across XML tags (e.g., `<t>{{</t><t>key</t><t>}}</t>`). If this happens, you'll need to first normalize the XML by joining adjacent text runs before replacing.
- The ZIP mimetype entry: if the original HWPX has a `mimetype` entry that must be stored uncompressed (like in ODF), preserve that. Check with: `python3 -c "import zipfile; z=zipfile.ZipFile('training_feedback_template.hwpx'); [print(i.filename, i.compress_type) for i in z.infolist()]"`

#### 5. Execute and verify
- Run: `python3 /root/fill_template.py`
- Check output for any warnings about unreplaced placeholders.
- Verify the output is a valid ZIP: `python3 -c "import zipfile; zipfile.ZipFile('/root/training_feedback_ready.hwpx').testzip()"`
- Spot-check a few XML files in the output to confirm values are correctly inserted.
- Specifically verify:
  - 참석자수 is digits only (no 명 or other suffix)
  - 만족도 follows the `X.X점 (5.0점 만점)` format
  - The overall opinion ends with `후속 심화반 검토 요망.`
  - No `{{` remains anywhere
  - Korean labels and static note lines are unchanged
  - Layout cache elements are removed from modified paragraphs

#### 6. Handle edge cases
- If placeholders are split across XML elements, parse the XML properly with `lxml.etree` or `xml.etree.ElementTree`, concatenate text within paragraph runs, perform replacement, then redistribute text back.
- If the original ZIP has specific compression settings per entry, preserve them.
- If `mimetype` must be first and uncompressed, handle that explicitly.

### Key constraints
- Output path must be exactly `/root/training_feedback_ready.hwpx`
- Must be a valid HWPX (ZIP) package
- Zero remaining `{{...}}` placeholders
- All Korean labels preserved
- Layout cache removed from edited paragraphs only

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