# Task Instruction

Complete the inventory report HWPX document by replacing all {{...}} placeholders with values from the JSON data file.

## Step-by-step Plan

### 1. Inspect the input files
- Read `/root/inventory_data.json` to understand the available keys and values.
- List the contents of `/root/inventory_report_template.hwpx` — it is a ZIP archive. Use `unzip -l` or Python's `zipfile` to list all entries.
- Extract the archive to a temporary directory (e.g., `/tmp/hwpx_work/`).
- Find and read the main content XML file(s). Typically this is `Contents/section0.xml` or similar. Read ALL XML files to find which ones contain `{{` placeholder text.

### 2. Understand the placeholder challenge
CRITICAL: In .hwpx files, a single placeholder like `{{report_date}}` is often split across multiple XML `<hp:t>` tags due to formatting runs, e.g.:
```xml
<hp:t>{{</hp:t></hp:run><hp:run>...<hp:t>report_date</hp:t></hp:run><hp:run>...<hp:t>}}</hp:t>
```
You MUST handle this. The proven approach is to work at the raw string level.

### 3. Write and run a Python script that does the following:

```python
import zipfile
import json
import os
import re
import shutil

# Load JSON data
with open('/root/inventory_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Flatten nested JSON if needed — inspect the structure first
# Build a flat key->value mapping for all placeholders

template_path = '/root/inventory_report_template.hwpx'
output_path = '/root/inventory_report_ready.hwpx'
work_dir = '/tmp/hwpx_work'

# Clean work dir
if os.path.exists(work_dir):
    shutil.rmtree(work_dir)
os.makedirs(work_dir)

# Extract
with zipfile.ZipFile(template_path, 'r') as zf:
    zf.extractall(work_dir)
    namelist = zf.namelist()

# Process each file in the archive
for name in namelist:
    filepath = os.path.join(work_dir, name)
    if not os.path.isfile(filepath):
        continue
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except (UnicodeDecodeError, IsADirectoryError):
        continue
    
    if '{{' not in content:
        continue
    
    original = content
    
    # Step A: Remove XML tags between {{ and }} to merge split placeholders
    # Repeatedly find {{ ... }} spans that may contain XML tags and strip internal tags
    def merge_split_placeholders(text):
        # Find all regions that start with {{ and end with }} but may have XML tags inside
        # Strategy: strip all XML tags from <hp:t> content perspective
        # Better: work on the raw string, find '{{' then find '}}' and remove XML tags between them
        result = text
        max_iterations = 50
        for _ in range(max_iterations):
            # Find a {{ that is followed by }} with possible XML tags in between
            match = re.search(r'\{\{((?:(?!\}\}).)*?)(<[^>]+>)((?:(?!\}\}).)*?\}\})', result, re.DOTALL)
            if not match:
                break
            # Remove the XML tag found between {{ and }}
            start, end = match.start(), match.end()
            inner = result[start:end]
            # Remove all XML tags from this span
            cleaned = re.sub(r'<[^>]+>', '', inner)
            result = result[:start] + cleaned + result[end:]
        return result
    
    content = merge_split_placeholders(content)
    
    # Step B: Now replace {{key}} placeholders with JSON values
    for key, value in data.items():
        if isinstance(value, dict):
            for subkey, subval in value.items():
                placeholder = '{{' + subkey + '}}'
                content = content.replace(placeholder, str(subval))
                # Also try dotted notation
                placeholder2 = '{{' + key + '.' + subkey + '}}'
                content = content.replace(placeholder2, str(subval))
        elif isinstance(value, list):
            # Handle list items - inspect and handle appropriately
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    for subkey, subval in item.items():
                        placeholder = '{{' + key + '[' + str(i) + '].' + subkey + '}}'
                        content = content.replace(placeholder, str(subval))
                        # Also try other naming patterns
                        placeholder2 = '{{' + subkey + '_' + str(i+1) + '}}'
                        content = content.replace(placeholder2, str(subval))
        else:
            placeholder = '{{' + key + '}}'
            content = content.replace(placeholder, str(value))
    
    # Step C: Remove stale layout cache (lineSegArray elements) from modified paragraphs
    # Remove all <hp:lineSegArray>...</hp:lineSegArray> from paragraphs that were modified
    if content != original:
        content = re.sub(r'<hp:lineSegArray>.*?</hp:lineSegArray>', '', content, flags=re.DOTALL)
    
    # Step D: Verify no {{...}} remain
    remaining = re.findall(r'\{\{.*?\}\}', content)
    if remaining:
        print(f'WARNING: Remaining placeholders in {name}: {remaining}')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# Repack as ZIP
with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for name in namelist:
        filepath = os.path.join(work_dir, name)
        if os.path.isfile(filepath):
            zf.write(filepath, name)

print('Done. Output written to', output_path)
```

### 4. IMPORTANT: Before running the full script, first inspect the data and template
- Print the JSON data structure to understand all keys and nesting.
- Print the raw XML content of files containing `{{` to understand the exact placeholder format and any split patterns.
- Adapt the placeholder replacement logic to match the ACTUAL placeholder names found in the XML and the ACTUAL key names in the JSON.
- Pay special attention to: nested JSON objects, list/array data that maps to table rows, numeric formatting, date formatting.

### 5. Validation
After generating the output:
- Use `unzip -l /root/inventory_report_ready.hwpx` to verify it's a valid ZIP.
- Extract and check the content XML(s) for any remaining `{{` patterns: `grep -r '{{' /tmp/hwpx_verify/`
- Verify Korean text is preserved by checking a few Korean strings are still present.
- Verify empty paragraphs are preserved (look for `<hp:p>` elements with empty or no `<hp:t>` content).
- Verify lineSegArray elements are removed from modified paragraphs.

### 6. Key constraints to remember
- Keep ALL Korean labels unchanged.
- Keep ALL empty paragraphs (spacing elements) in the document.
- Remove ALL `{{...}}` placeholders — none may remain.
- Remove `lineSegArray` layout cache from any paragraph whose text was modified.
- Output must be a valid .hwpx (ZIP) package at `/root/inventory_report_ready.hwpx`.
- Do NOT remove or alter `lineSegArray` from paragraphs that were NOT modified (to minimize changes).

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