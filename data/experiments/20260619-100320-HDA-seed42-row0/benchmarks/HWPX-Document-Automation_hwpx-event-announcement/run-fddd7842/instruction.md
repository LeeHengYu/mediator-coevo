# Task Instruction

Complete the following task to fill an HWPX event announcement template with data from a JSON file.

## Goal
Replace all `{{...}}` placeholders in `event_announcement_template.hwpx` with values from `event_data.json`, then save the result to `/root/event_announcement_ready.hwpx`.

## Step-by-step Instructions

### Step 1: Inspect the input files
1. List the files in the current working directory to locate `event_announcement_template.hwpx` and `event_data.json`.
2. Read and display the contents of `event_data.json` to understand all available keys and values.

### Step 2: Extract and inspect the HWPX template
1. Create a temporary working directory, e.g., `/tmp/hwpx_work`.
2. Unzip `event_announcement_template.hwpx` into that directory (it is a ZIP archive).
3. List the full directory tree to understand the package structure.
4. Read `Contents/section0.xml` (and any other XML files under `Contents/`) to find all `{{...}}` placeholders. Also check `Contents/content.hpf` or any other XML files for placeholders.
5. List every unique `{{...}}` placeholder you find across all files. Confirm each one has a matching key in `event_data.json`.

### Step 3: Write and run a Python script to perform the replacement
Write a single Python script (`/tmp/fill_template.py`) that does the following:

```python
import json, os, shutil, zipfile, re
from lxml import etree

# Paths
template_path = '<path to event_announcement_template.hwpx>'
json_path = '<path to event_data.json>'
output_path = '/root/event_announcement_ready.hwpx'
extract_dir = '/tmp/hwpx_extracted'

# Clean previous runs
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)

# Load JSON data
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract HWPX
with zipfile.ZipFile(template_path, 'r') as zf:
    zf.extractall(extract_dir)

# Walk through ALL files in the extracted archive
for root, dirs, files in os.walk(extract_dir):
    for fname in files:
        fpath = os.path.join(root, fname)
        # Only process text/xml files
        if not fname.endswith(('.xml', '.hpf', '.rels')):
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace all {{key}} placeholders with values from JSON
        for key, value in data.items():
            content = content.replace('{{' + key + '}}', str(value))
        
        # Also handle placeholders that may be split across XML tags:
        # Rebuild by stripping tags, replacing, then... 
        # Actually, do a second pass: find any remaining {{...}} by
        # checking if tags split them. If any remain after simple replacement,
        # handle split-tag case.
        
        if content != original_content or '{{' in content:
            # Write back the text-replaced content
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Now parse as XML to handle split-tag placeholders and remove layout cache
            # Re-read to parse
            try:
                # Register namespaces to preserve prefixes
                # First, collect all namespaces from the file
                ns_map = {}
                for event, elem in etree.iterparse(fpath, events=('start-ns',)):
                    prefix, uri = elem
                    if prefix:
                        ns_map[prefix] = uri
                
                for prefix, uri in ns_map.items():
                    etree.register_namespace(prefix, uri)
                
                tree = etree.parse(fpath)
                root_elem = tree.getroot()
                
                # Collect all namespace URIs for hp prefix
                hp_ns = ns_map.get('hp', None)
                
                # Handle split-tag placeholders:
                # Find all paragraph elements and concatenate their text runs
                if hp_ns:
                    # Find paragraphs
                    for p in root_elem.iter('{%s}p' % hp_ns):
                        # Collect all <hp:t> elements
                        t_elements = list(p.iter('{%s}t' % hp_ns))
                        if not t_elements:
                            continue
                        # Concatenate all text
                        full_text = ''.join((t.text or '') for t in t_elements)
                        # Check if there's a remaining placeholder
                        if '{{' in full_text and '}}' in full_text:
                            # Replace placeholders
                            for key, value in data.items():
                                full_text = full_text.replace('{{' + key + '}}', str(value))
                            # Put all text in first <hp:t>, clear the rest
                            if t_elements:
                                t_elements[0].text = full_text
                                for t in t_elements[1:]:
                                    t.text = ''
                        
                        # Check if this paragraph was modified - remove lineSegArray
                        # (layout cache) to prevent rendering issues
                        p_text = ''.join((t.text or '') for t in p.iter('{%s}t' % hp_ns))
                        # Remove lineSegArray elements from modified paragraphs
                        for lsa in p.findall('.//{%s}lineSegArray' % hp_ns):
                            lsa.getparent().remove(lsa)
                
                # Write back
                tree.write(fpath, xml_declaration=True, encoding='UTF-8')
            except Exception as e:
                print(f'XML parse warning for {fpath}: {e}')
                # If XML parsing fails, the text replacement is still in place

# Check for any remaining placeholders
remaining = []
for root, dirs, files in os.walk(extract_dir):
    for fname in files:
        fpath = os.path.join(root, fname)
        if not fname.endswith(('.xml', '.hpf', '.rels')):
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        found = re.findall(r'\{\{.*?\}\}', content)
        if found:
            remaining.append((fpath, found))

if remaining:
    print('WARNING: Remaining placeholders found:')
    for fpath, phs in remaining:
        print(f'  {fpath}: {phs}')
else:
    print('All placeholders replaced successfully.')

# Re-package as HWPX (ZIP)
if os.path.exists(output_path):
    os.remove(output_path)

with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(extract_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            arcname = os.path.relpath(fpath, extract_dir)
            # mimetype should be stored, not deflated (if present)
            if fname == 'mimetype':
                zf.write(fpath, arcname, compress_type=zipfile.ZIP_STORED)
            else:
                zf.write(fpath, arcname)

print(f'Output saved to {output_path}')
```

Adapt the paths based on what you find in Step 1. The script above is a template — adjust namespace handling and file paths as needed based on actual inspection.

### Step 4: Run the script and validate
1. Run the Python script.
2. Verify the output:
   - Unzip `/root/event_announcement_ready.hwpx` to a temp location.
   - Read `Contents/section0.xml` and confirm:
     a. No `{{...}}` placeholders remain anywhere.
     b. All JSON values appear correctly in the document.
     c. Korean labels and static note lines are preserved unchanged.
     d. No `<hp:lineSegArray>` (or equivalent `lineSegArray` in any namespace) elements exist in paragraphs that were modified.
   - Confirm the ZIP structure matches the original template's structure.
3. Print a summary of replaced values for confirmation.

### Step 5: Final check
- Run `zipfile.is_zipfile('/root/event_announcement_ready.hwpx')` to confirm it's a valid ZIP.
- Grep all XML files in the output for `{{` to absolutely confirm no placeholders remain.
- If any issues are found, fix and re-run.

## Critical Requirements
- **Namespace preservation**: Register all XML namespaces before parsing to avoid prefix loss.
- **Split-tag handling**: Placeholders like `{{event_name}}` may be split across multiple `<hp:t>` tags due to formatting. Concatenate text runs within a paragraph, perform replacement, then redistribute.
- **Layout cache removal**: Remove ALL `<hp:lineSegArray>` elements from any paragraph (`<hp:p>`) that had text modifications. This prevents character overlap rendering issues.
- **ZIP packaging**: Ensure the archive root contains the same structure as the original (e.g., `mimetype` at root level, `Contents/` directory, etc.). Store `mimetype` without compression if it exists.
- **No placeholder residue**: The final output must contain zero `{{...}}` patterns in any file.

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