# Task Instruction

Execute the following steps to produce `/root/supplier_contact_ready.hwpx`.

## Step 1 – Inspect the workspace
```bash
ls /root/
file /root/supplier_contact_template.hwpx
cat /root/supplier_contact.json
```
Understand the JSON keys and values.

## Step 2 – Unpack the HWPX template
```bash
mkdir -p /tmp/hwpx_work
cd /tmp/hwpx_work
unzip -o /root/supplier_contact_template.hwpx -d template
```
List the extracted tree (`find template -type f`) and inspect `template/Contents/section0.xml` (or whichever section file exists) to see the XML structure and all `{{…}}` placeholders.

## Step 3 – Write and run the Python transformation script

Create `/tmp/hwpx_work/transform.py` with the following logic:

```python
import json, os, shutil, zipfile
from lxml import etree

# ── paths ──
TEMPLATE_DIR = '/tmp/hwpx_work/template'
JSON_PATH = '/root/supplier_contact.json'
OUTPUT_PATH = '/root/supplier_contact_ready.hwpx'

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ── namespace map (extend if the XML uses more) ──
NS = {
    'hp':  'http://www.hancom.co.kr/hwpml/2011/paragraph',
    'hs':  'http://www.hancom.co.kr/hwpml/2011/section',
    'hc':  'http://www.hancom.co.kr/hwpml/2011/common',
    'hw':  'http://www.hancom.co.kr/hwpml/2011/head',
    'hpb': 'urn:hancom:hwpml:2011:paragraph:body',
    'config': 'urn:hancom:hwpml:2011:config',
    'para': 'http://www.hancom.co.kr/hwpml/2011/paragraph',
}

# Discover actual namespaces from the section XML and register them all
section_files = []
for root_dir, dirs, files in os.walk(TEMPLATE_DIR):
    for fn in files:
        if fn.endswith('.xml'):
            full = os.path.join(root_dir, fn)
            # Check if it is a section file (contains paragraphs)
            section_files.append(full)

def register_namespaces(xml_path):
    """Parse once just to grab namespace prefixes, then register them."""
    events = etree.iterparse(xml_path, events=['start-ns'])
    nsmap = {}
    for event, (prefix, uri) in events:
        if prefix:
            nsmap[prefix] = uri
    for prefix, uri in nsmap.items():
        etree.register_namespace(prefix, uri)
    return nsmap

def process_section(xml_path, replacements):
    nsmap = register_namespaces(xml_path)
    tree = etree.parse(xml_path)
    root = tree.getroot()

    # Resolve hp namespace URI from the file itself
    hp_uri = nsmap.get('hp') or NS['hp']
    p_tag   = f'{{{hp_uri}}}p'
    run_tag = f'{{{hp_uri}}}run'
    t_tag   = f'{{{hp_uri}}}t'
    lsa_tag = f'{{{hp_uri}}}lineSegArray'

    modified_paragraphs = []

    for p_elem in root.iter(p_tag):
        # Collect all <hp:run> elements and their <hp:t> children
        runs = list(p_elem.iter(run_tag))
        if not runs:
            continue

        # Build the full paragraph text by concatenating all <hp:t> texts
        t_elements = []
        for run in runs:
            for t in run.iter(t_tag):
                t_elements.append(t)

        full_text = ''.join((t.text or '') for t in t_elements)

        if '{{' not in full_text:
            continue

        # Perform replacements on the concatenated text
        new_text = full_text
        for key, value in replacements.items():
            placeholder = '{{' + key + '}}'
            new_text = new_text.replace(placeholder, str(value))

        if new_text == full_text:
            continue

        # Redistribute the new text: put everything in the first <hp:t>,
        # clear the rest
        if t_elements:
            t_elements[0].text = new_text
            for t in t_elements[1:]:
                t.text = ''

        modified_paragraphs.append(p_elem)

    # Remove layout cache from modified paragraphs
    for p_elem in modified_paragraphs:
        for lsa in list(p_elem.iter(lsa_tag)):
            lsa.getparent().remove(lsa)

    tree.write(xml_path, xml_declaration=True, encoding='UTF-8')

# ── process every XML that might contain placeholders ──
for xml_path in section_files:
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if '{{' in content:
            process_section(xml_path, data)
    except Exception as e:
        print(f'Skipping {xml_path}: {e}')

# ── repackage as HWPX ──
# mimetype must be first entry, stored uncompressed
mimetype_path = os.path.join(TEMPLATE_DIR, 'mimetype')
with zipfile.ZipFile(OUTPUT_PATH, 'w') as zf:
    # 1. mimetype – uncompressed
    if os.path.exists(mimetype_path):
        zf.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)

    # 2. everything else – deflated
    for dirpath, dirnames, filenames in os.walk(TEMPLATE_DIR):
        for filename in filenames:
            abs_path = os.path.join(dirpath, filename)
            arc_name = os.path.relpath(abs_path, TEMPLATE_DIR)
            if arc_name == 'mimetype':
                continue
            zf.write(abs_path, arc_name, compress_type=zipfile.ZIP_DEFLATED)

print('Done – wrote', OUTPUT_PATH)
```

Run it:
```bash
cd /tmp/hwpx_work && python3 transform.py
```

## Step 4 – Validate the output

```bash
# 1. Confirm it is a valid ZIP
unzip -t /root/supplier_contact_ready.hwpx

# 2. Confirm no remaining placeholders
unzip -p /root/supplier_contact_ready.hwpx | grep -c '{{' || true
# Expected: 0

# 3. Spot-check section XML for replaced values and preserved Korean labels
mkdir -p /tmp/hwpx_verify
unzip -o /root/supplier_contact_ready.hwpx -d /tmp/hwpx_verify
cat /tmp/hwpx_verify/Contents/section0.xml | python3 -c "import sys; d=sys.stdin.read(); assert '{{' not in d, 'Placeholders remain!'; print('No placeholders remain. OK')"

# 4. Verify Korean labels are still present (grep for a couple)
grep -c '회사' /tmp/hwpx_verify/Contents/section0.xml || echo 'WARNING: Korean labels may be missing'

# 5. Check mimetype is first entry
python3 -c "
import zipfile
zf = zipfile.ZipFile('/root/supplier_contact_ready.hwpx')
first = zf.namelist()[0]
assert first == 'mimetype', f'First entry is {first}, not mimetype'
print('mimetype is first entry. OK')
"
```

If any validation fails, inspect the extracted XML, diagnose, fix the script, and re-run until all checks pass.

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