# Task Instruction

Execute the following steps to fill in the training feedback HWPX template and produce `/root/training_feedback_ready.hwpx`.

### Step 1 — Inspect inputs
1. `cat /root/training_feedback.json` — note every key/value.
2. `cp /root/training_feedback_template.hwpx /tmp/template.hwpx`
3. `mkdir -p /tmp/hwpx_work && cd /tmp/hwpx_work && unzip -o /tmp/template.hwpx`
4. List the extracted tree: `find . -type f`
5. Read every `section*.xml` file (likely `Contents/section0.xml`) and any other XML files that could contain `{{` placeholders: `grep -rl '{{' .`

### Step 2 — Understand the XML structure
Open each file containing `{{` and study:
- The namespace declarations (especially `hp:` prefix and its URI).
- How `<hp:p>` paragraphs contain `<hp:run>` → `<hp:t>` text nodes.
- Whether any `{{placeholder}}` is split across multiple `<hp:t>` nodes within one `<hp:run>` or across multiple `<hp:run>` elements.
- The presence of `<hp:lineSegArray>` (or `<hp:linesegarray>` — check case) inside `<hp:p>` elements. These are layout-cache elements that must be removed from any paragraph you modify.

### Step 3 — Write and run a Python script
Write `/tmp/fill_template.py` that does the following:

```python
import json, os, re, shutil, zipfile
from lxml import etree

# --- paths ---
json_path = '/root/training_feedback.json'
work_dir = '/tmp/hwpx_work'
output_path = '/root/training_feedback_ready.hwpx'

# --- load JSON values ---
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# --- value transforms ---
# 참석자수 → digits only
for k in data:
    if '참석자수' in k or 'attendee' in k.lower() or k == '참석자수':
        data[k] = re.sub(r'[^0-9]', '', str(data[k]))

# 만족도 → "X.X점 (5.0점 만점)" style
for k in data:
    if '만족도' in k or 'satisfaction' in k.lower() or k == '만족도':
        score = str(data[k])
        # extract numeric part
        m = re.search(r'[\d.]+', score)
        if m:
            data[k] = f"{m.group()}점 (5.0점 만점)"

# --- find all XML files with {{ ---
xml_files = []
for root, dirs, files in os.walk(work_dir):
    for fn in files:
        fp = os.path.join(root, fn)
        if fn.endswith('.xml'):
            with open(fp, 'r', encoding='utf-8') as fh:
                if '{{' in fh.read():
                    xml_files.append(fp)

print('Files with placeholders:', xml_files)

# --- process each XML file ---
for xml_file in xml_files:
    tree = etree.parse(xml_file)
    root_el = tree.getroot()
    nsmap = root_el.nsmap
    # build ns dict for findall; handle default ns
    ns = {}
    for prefix, uri in nsmap.items():
        if prefix is not None:
            ns[prefix] = uri

    # Find all <hp:p> paragraphs (try multiple namespace approaches)
    paragraphs = tree.xpath('//*[local-name()="p"]')

    for para in paragraphs:
        # Collect all <hp:t> text nodes in this paragraph
        t_nodes = para.xpath('.//*[local-name()="t"]')
        if not t_nodes:
            continue

        # Aggregate full text to check for placeholders
        full_text = ''.join((t.text or '') for t in t_nodes)
        if '{{' not in full_text:
            continue

        # --- Replace placeholders ---
        # Strategy: consolidate all text into first <t>, clear the rest
        replaced = full_text
        for key, val in data.items():
            replaced = replaced.replace('{{' + key + '}}', str(val))

        # Handle overall-opinion: append follow-up sentence
        # The instruction says: append '후속 심화반 검토 요망.' after the provided comment
        # We look for the overall opinion value and append if this paragraph contains it
        # We'll do this by checking if any opinion/comment value from JSON is in the replaced text
        # More robust: check all data values that look like opinion/comment text
        # We handle this after all replacements

        # Set first t_node text, clear others
        t_nodes[0].text = replaced
        for t in t_nodes[1:]:
            t.text = ''

        # Remove layout cache: <hp:lineSegArray> or any element with local-name 'lineSegArray' or 'linesegarray'
        for cache_el in para.xpath('.//*[local-name()="lineSegArray" or local-name()="linesegarray" or local-name()="lineSegArray"]'):
            cache_el.getparent().remove(cache_el)

    # --- Second pass: append follow-up sentence to overall opinion ---
    # Re-scan paragraphs for the opinion value
    # Identify which JSON key is the overall opinion
    # The instruction says "final overall-opinion sentence" — look for the comment value
    # and append '후속 심화반 검토 요망.' if not already present
    all_paras = tree.xpath('//*[local-name()="p"]')
    for para in all_paras:
        t_nodes = para.xpath('.//*[local-name()="t"]')
        full_text = ''.join((t.text or '') for t in t_nodes)
        # Find the opinion/comment value from JSON
        for key, val in data.items():
            sval = str(val)
            if len(sval) > 10 and sval in full_text:  # likely the comment field
                suffix = ' 후속 심화반 검토 요망.'
                if '후속 심화반 검토 요망' not in full_text:
                    t_nodes[0].text = full_text + suffix
                    for t in t_nodes[1:]:
                        t.text = ''
                    # Remove layout cache again
                    for cache_el in para.xpath('.//*[local-name()="lineSegArray" or local-name()="linesegarray"]'):
                        cache_el.getparent().remove(cache_el)

    # Save
    tree.write(xml_file, xml_declaration=True, encoding='UTF-8', pretty_print=False)

# --- Verify no {{ remain ---
for xml_file in xml_files:
    with open(xml_file, 'r', encoding='utf-8') as f:
        content = f.read()
        remaining = re.findall(r'\{\{[^}]+\}\}', content)
        if remaining:
            print(f'WARNING: remaining placeholders in {xml_file}: {remaining}')
        else:
            print(f'OK: no placeholders remain in {xml_file}')

# --- Repack as .hwpx (ZIP) preserving structure ---
if os.path.exists(output_path):
    os.remove(output_path)

# Walk the work_dir and add files with correct relative paths
with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for dirpath, dirnames, filenames in os.walk(work_dir):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            arcname = os.path.relpath(full, work_dir)
            # mimetype should be stored uncompressed if present
            if fn == 'mimetype':
                zf.write(full, arcname, compress_type=zipfile.ZIP_STORED)
            else:
                zf.write(full, arcname)

print(f'Output written to {output_path}')
```

Run: `cd /tmp && python3 fill_template.py`

### Step 4 — Validate
1. Verify the output exists: `ls -la /root/training_feedback_ready.hwpx`
2. Unzip to a temp dir and check no `{{` remain: `mkdir -p /tmp/verify && cd /tmp/verify && unzip -o /root/training_feedback_ready.hwpx && grep -r '{{' . || echo 'No placeholders found — GOOD'`
3. Verify 참석자수 is digits only: `grep -oP '참석자수.*?<' /tmp/verify/Contents/section*.xml` (should show only digits in the value)
4. Verify 만족도 has the "X.X점 (5.0점 만점)" format: `grep '만족도' /tmp/verify/Contents/section*.xml`
5. Verify the follow-up sentence exists: `grep '후속 심화반 검토 요망' /tmp/verify/Contents/section*.xml`
6. Verify no lineSegArray in modified paragraphs: run a quick Python check that for every `<hp:p>` whose `<hp:t>` text was changed, there is no `lineSegArray` child.
7. Run the verifier if available: `cd /root && python -m pytest test_output*.py -v` or equivalent.

### Step 5 — Fix issues
If any placeholder remains, it's likely split across `<hp:t>` nodes in a way the aggregation missed (e.g., across different `<hp:run>` elements). Re-examine the raw XML, adjust the consolidation logic, and re-run.

If the verifier fails on layout cache, double-check the element local name (case-sensitive) and ensure removal covers all modified paragraphs.

### Important notes
- The script above is a starting template. After Step 1 inspection, **adapt the JSON keys and the opinion-detection logic** to match the actual key names found in `training_feedback.json`.
- The follow-up sentence `후속 심화반 검토 요망.` must be appended (with a space before it) to the overall opinion text, not placed in a separate paragraph.
- Korean labels and the static note line must remain untouched.
- The final file must be a valid ZIP-based .hwpx package.

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