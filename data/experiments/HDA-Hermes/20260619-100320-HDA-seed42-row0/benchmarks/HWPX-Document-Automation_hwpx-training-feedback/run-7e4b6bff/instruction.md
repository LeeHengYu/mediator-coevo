# Task Instruction

Execute the following Python script to fill in the training feedback template and produce the output HWPX file.

```python
import zipfile
import os
import json
import re
import shutil

# Paths
template_path = '/root/training_feedback_template.hwpx'
json_path = '/root/training_feedback.json'
output_path = '/root/training_feedback_ready.hwpx'
extract_dir = '/tmp/hwpx_extract'

# Clean up any previous extraction
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)
os.makedirs(extract_dir)

# Step 1: Read the JSON data
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print('JSON data:')
print(json.dumps(data, ensure_ascii=False, indent=2))

# Step 2: Extract the HWPX (ZIP) archive
with zipfile.ZipFile(template_path, 'r') as zf:
    zf.extractall(extract_dir)
    namelist = zf.namelist()

print('\nFiles in HWPX:')
for name in namelist:
    print(f'  {name}')

# Step 3: Find all XML files that might contain content
xml_files = []
for root, dirs, files in os.walk(extract_dir):
    for fname in files:
        if fname.endswith('.xml'):
            fpath = os.path.join(root, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            if '{{' in content or 'hp:t' in content:
                xml_files.append(fpath)
                print(f'\nContent file: {os.path.relpath(fpath, extract_dir)}')

# Step 4: Build replacement map from JSON
# Flatten nested JSON into a simple key->value map
def flatten_json(obj, prefix=''):
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                items.update(flatten_json(v, prefix))
            else:
                items[k] = str(v)
    elif isinstance(obj, list):
        for item in obj:
            items.update(flatten_json(item, prefix))
    return items

replacements = flatten_json(data)
print('\nFlattened replacements:')
for k, v in replacements.items():
    print(f'  {k} -> {v}')

# Step 5: Apply special transformations
# 참석자수: convert to digits only
if '참석자수' in replacements:
    val = replacements['참석자수']
    digits = re.sub(r'[^0-9]', '', val)
    if digits:
        replacements['참석자수'] = digits
    print(f'  참석자수 transformed to: {replacements["참석자수"]}')

# 만족도: rewrite as "X.X점 (5.0점 만점)" style
if '만족도' in replacements:
    val = replacements['만족도']
    # Extract numeric score
    score_match = re.search(r'([0-9]+\.?[0-9]*)', val)
    if score_match:
        score = score_match.group(1)
        # Ensure one decimal place
        if '.' not in score:
            score = score + '.0'
        replacements['만족도'] = f'{score}점 (5.0점 만점)'
    print(f'  만족도 transformed to: {replacements["만족도"]}')

# 종합의견 / overall opinion: append 후속 심화반 검토 요망.
for key in ['종합의견', '종합 의견', 'overall_opinion', 'comment', '의견']:
    if key in replacements:
        val = replacements[key].rstrip()
        if not val.endswith('후속 심화반 검토 요망.'):
            if val and not val.endswith(' '):
                val = val + ' '
            val = val + '후속 심화반 검토 요망.'
        replacements[key] = val
        print(f'  {key} transformed to: {replacements[key]}')

print('\nFinal replacements:')
for k, v in replacements.items():
    print(f'  {k} -> {v}')

# Step 6: Process each XML file
for fpath in xml_files:
    with open(fpath, 'r', encoding='utf-8') as f:
        xml = f.read()

    # 6a: Consolidate split placeholder tags within <hp:run> elements
    # This merges text across multiple <hp:t> tags within the same <hp:run>
    def consolidate_run(match):
        run_xml = match.group(0)
        # Extract all <hp:t ...>text</hp:t> contents
        t_pattern = r'<hp:t[^>]*>(.*?)</hp:t>'
        texts = re.findall(t_pattern, run_xml, re.DOTALL)
        if len(texts) > 1:
            merged = ''.join(texts)
            # Check if merged text contains placeholder fragments
            if '{{' in merged or '}}' in merged:
                # Remove all <hp:t> elements
                cleaned = re.sub(t_pattern, '', run_xml, flags=re.DOTALL)
                # Find position to insert the single merged <hp:t>
                # Insert before </hp:run>
                cleaned = cleaned.replace('</hp:run>', f'<hp:t>{merged}</hp:t></hp:run>')
                return cleaned
        return run_xml

    xml = re.sub(r'<hp:run[^>]*>.*?</hp:run>', consolidate_run, xml, flags=re.DOTALL)

    # 6b: Also consolidate within <hp:p> across runs if placeholders span runs
    def consolidate_paragraph(match):
        p_xml = match.group(0)
        # Get all text content
        t_pattern = r'<hp:t[^>]*>(.*?)</hp:t>'
        texts = re.findall(t_pattern, p_xml, re.DOTALL)
        full_text = ''.join(texts)
        if '{{' in full_text and '}}' in full_text:
            # Check if any single <hp:t> doesn't have a complete placeholder
            # but the combined text does
            has_split = False
            for t in texts:
                if ('{{' in t and '}}' not in t) or ('}}' in t and '{{' not in t):
                    has_split = True
                    break
                # Also check for partial like {{ with rest in next tag
                opens = t.count('{{')
                closes = t.count('}}')
                if opens != closes:
                    has_split = True
                    break
            if has_split:
                # Merge all runs' text into the first <hp:t>, remove others
                first = True
                def replace_t(m):
                    nonlocal first
                    if first:
                        first = True  # keep replacing until we find first
                result = p_xml
                # Remove all <hp:t> tags and their content
                t_tags = list(re.finditer(t_pattern, result, re.DOTALL))
                if t_tags:
                    # Replace first with merged, remove rest
                    for i, tm in enumerate(reversed(t_tags)):
                        if i == len(t_tags) - 1:  # This is actually the first match (reversed)
                            result = result[:tm.start()] + f'<hp:t>{full_text}</hp:t>' + result[tm.end():]
                        else:
                            result = result[:tm.start()] + result[tm.end():]
                return result
        return p_xml

    xml = re.sub(r'<hp:p\b[^>]*>.*?</hp:p>', consolidate_paragraph, xml, flags=re.DOTALL)

    # 6c: Replace all {{key}} placeholders
    modified = False
    for key, value in replacements.items():
        placeholder = '{{' + key + '}}'
        if placeholder in xml:
            xml = xml.replace(placeholder, value)
            modified = True
            print(f'  Replaced {placeholder} in {os.path.relpath(fpath, extract_dir)}')

    # Also try with spaces inside braces (e.g., {{ key }})
    for key, value in replacements.items():
        pattern = r'\{\{\s*' + re.escape(key) + r'\s*\}\}'
        new_xml = re.sub(pattern, value, xml)
        if new_xml != xml:
            xml = new_xml
            modified = True
            print(f'  Replaced (with spaces) {key} in {os.path.relpath(fpath, extract_dir)}')

    # 6d: Remove <hp:lineSegArray> from modified paragraphs (or all paragraphs to be safe)
    # Use regex to remove ALL lineSegArray elements
    xml_before = xml
    xml = re.sub(r'<hp:lineSegArray[^>]*/>', '', xml)  # self-closing
    xml = re.sub(r'<hp:lineSegArray[^>]*>.*?</hp:lineSegArray>', '', xml, flags=re.DOTALL)
    # Also handle namespace variations
    xml = re.sub(r'<lineSegArray[^>]*/>', '', xml)
    xml = re.sub(r'<lineSegArray[^>]*>.*?</lineSegArray>', '', xml, flags=re.DOTALL)
    if xml != xml_before:
        print(f'  Removed lineSegArray elements from {os.path.relpath(fpath, extract_dir)}')

    # 6e: Check for any remaining placeholders
    remaining = re.findall(r'\{\{.*?\}\}', xml)
    if remaining:
        print(f'  WARNING: Remaining placeholders in {os.path.relpath(fpath, extract_dir)}: {remaining}')

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(xml)

# Step 7: Re-zip as HWPX
# mimetype must be first entry, stored (not compressed)
if os.path.exists(output_path):
    os.remove(output_path)

mimetype_path = os.path.join(extract_dir, 'mimetype')
with zipfile.ZipFile(output_path, 'w') as zout:
    # Write mimetype first with STORED compression
    if os.path.exists(mimetype_path):
        zout.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)
    
    for root, dirs, files in os.walk(extract_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            arcname = os.path.relpath(fpath, extract_dir)
            if arcname == 'mimetype':
                continue  # already added
            zout.write(fpath, arcname, compress_type=zipfile.ZIP_DEFLATED)

print(f'\nOutput written to {output_path}')

# Step 8: Verify output
with zipfile.ZipFile(output_path, 'r') as zf:
    for name in zf.namelist():
        if name.endswith('.xml'):
            content = zf.read(name).decode('utf-8')
            remaining = re.findall(r'\{\{.*?\}\}', content)
            if remaining:
                print(f'VERIFICATION FAIL: {name} still has placeholders: {remaining}')
            if 'lineSegArray' in content:
                print(f'VERIFICATION WARN: {name} still has lineSegArray elements')
    print('Verification complete.')
```

After running the script, verify:
1. The output file exists at `/root/training_feedback_ready.hwpx`
2. No `{{...}}` placeholders remain in any XML file within the HWPX
3. No `lineSegArray` elements remain in the output
4. The mimetype file is the first entry in the ZIP and is stored (not deflated)
5. 참석자수 is digits only
6. 만족도 follows the `X.X점 (5.0점 만점)` format
7. The overall opinion sentence ends with `후속 심화반 검토 요망.`

If the script prints any WARNING about remaining placeholders, inspect the XML to understand the split pattern and fix the consolidation logic. The most common issue is placeholders split across `<hp:run>` boundaries or across `<hp:t>` tags with intervening formatting elements.

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