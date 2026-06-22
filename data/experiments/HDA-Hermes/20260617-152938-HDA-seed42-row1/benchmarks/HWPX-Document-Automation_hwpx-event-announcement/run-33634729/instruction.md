# Task Instruction

Execute the following steps to prepare the event announcement HWPX document:

1. **Inspect the workspace.** List files in the current working directory to locate `event_announcement_template.hwpx` and `event_data.json`. Also check for any `test_output.py` or verifier script.

2. **Read the JSON data.** Load `event_data.json` and print its contents so you know every key-value pair available for substitution.

3. **Explore the HWPX package.** A `.hwpx` file is a ZIP archive. List all entries inside `event_announcement_template.hwpx`. Identify the main content XML file (typically `Contents/section0.xml` or similar). Extract and print that XML to understand the document structure, placeholder locations, and layout-cache elements.

4. **Write and run a Python script** (`fill_template.py`) that does the following:

```python
import json, zipfile, os, re, shutil, copy
from lxml import etree

TEMPLATE = 'event_announcement_template.hwpx'
DATA_FILE = 'event_data.json'
OUTPUT = '/root/event_announcement_ready.hwpx'

# Load JSON data
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Copy template to output location first
shutil.copy2(TEMPLATE, OUTPUT)

# We will edit XML files inside the ZIP in-place
# Read all entries from the template
with zipfile.ZipFile(TEMPLATE, 'r') as zin:
    entries = {name: zin.read(name) for name in zin.namelist()}

# Find XML content files that might contain placeholders
xml_files = [n for n in entries if n.endswith('.xml')]

for xml_name in xml_files:
    raw = entries[xml_name]
    # Quick check if any placeholder pattern exists
    if b'{{' not in raw:
        continue
    
    tree = etree.fromstring(raw)
    nsmap = tree.nsmap
    # Collect all namespaces for xpath
    ns = {k: v for k, v in nsmap.items() if k is not None}
    if None in nsmap:
        ns['default'] = nsmap[None]
    
    # Find all paragraph-like elements. In HWPX, paragraphs are typically
    # <hp:p> elements containing <hp:run> with <hp:t> text nodes.
    # We need to handle placeholders split across multiple <hp:t> elements.
    
    # Strategy: for each paragraph element, concatenate all text content,
    # perform replacements, then redistribute text back.
    
    # Find all 'p' elements (paragraphs)
    # Try multiple namespace prefixes
    p_elements = tree.iter()
    
    # Gather paragraphs - look for elements whose local name is 'p'
    paragraphs = [el for el in tree.iter() if etree.QName(el.tag).localname == 'p']
    
    for p in paragraphs:
        # Find all text elements (local name 't') within this paragraph
        t_elements = [el for el in p.iter() if etree.QName(el.tag).localname == 't']
        if not t_elements:
            continue
        
        # Concatenate all text
        full_text = ''.join((t.text or '') for t in t_elements)
        
        # Check if any placeholder exists
        if '{{' not in full_text:
            continue
        
        # Replace all placeholders
        new_text = full_text
        for key, value in data.items():
            placeholder = '{{' + key + '}}'
            new_text = new_text.replace(placeholder, str(value))
        
        # Also catch any remaining placeholders with regex (e.g., nested or unusual keys)
        # This is a safety net
        remaining = re.findall(r'\{\{(.+?)\}\}', new_text)
        if remaining:
            print(f'WARNING: unresolved placeholders in paragraph: {remaining}')
        
        if new_text != full_text:
            # Put all text into the first <t> element, clear the rest
            t_elements[0].text = new_text
            for t in t_elements[1:]:
                t.text = ''
            
            # Remove layout cache (lineSegArray) from this paragraph
            # lineSegArray elements can cause overlapping characters
            for child in list(p):
                local = etree.QName(child.tag).localname
                if local in ('lineSegArray', 'linesegarray'):
                    p.remove(child)
    
    # Check for any remaining placeholders in the entire XML
    serialized = etree.tostring(tree, xml_declaration=True, encoding='UTF-8')
    if b'{{' in serialized:
        print(f'WARNING: remaining placeholders in {xml_name}')
    
    entries[xml_name] = serialized

# Write the output HWPX
with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zout:
    for name, content in entries.items():
        zout.writestr(name, content)

print(f'Output written to {OUTPUT}')

# Verify: re-open and check for placeholders
with zipfile.ZipFile(OUTPUT, 'r') as z:
    for name in z.namelist():
        content = z.read(name)
        if b'{{' in content:
            print(f'FAIL: placeholder still in {name}')
        else:
            if name.endswith('.xml'):
                print(f'OK: {name} - no placeholders')

print('Done.')
```

5. **Run the script:** `python fill_template.py`

6. **Verify the output:**
   - Confirm `/root/event_announcement_ready.hwpx` exists and is a valid ZIP.
   - Open it with `zipfile` and check that no `{{` remains in any entry.
   - If `test_output.py` exists, run `python -m pytest test_output.py -v` and confirm all tests pass.

7. **Debug if needed:**
   - If placeholders remain, inspect which XML file and paragraph still contains them. The issue is likely placeholders split across runs in a way the script didn't handle. In that case, also check for `<hp:t>` elements nested inside `<hp:run>` inside `<hp:run>` or other nesting, and adjust the concatenation logic.
   - If layout cache tests fail, ensure you're removing all `lineSegArray` elements from modified paragraphs (check the exact namespace-qualified tag name by printing it).
   - Make sure the output is written as a proper ZIP (not nested ZIP-in-ZIP from the `shutil.copy2` that was overwritten).

The key technique (proven in prior successful runs): concatenate all `<t>` text within a paragraph before doing placeholder replacement, put the result in the first `<t>`, empty the rest, and strip `lineSegArray` layout cache from modified paragraphs.

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