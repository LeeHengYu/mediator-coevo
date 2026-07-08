# Task Instruction

Complete the following task to fill in a training feedback HWPX document from JSON data.

## Goal
Fill in `training_feedback_template.hwpx` using values from `training_feedback.json`, saving the result to `/root/training_feedback_ready.hwpx`.

## Steps

### 1. Inspect the input files
- Find the template and JSON files in the task directory. They should be at `/root/training_feedback_template.hwpx` and `/root/training_feedback.json` (or in a subdirectory — use `find / -name 'training_feedback*' 2>/dev/null` to locate them).
- Read the JSON file to understand all key-value pairs.
- Unzip the HWPX template to a temporary directory (it's a ZIP archive): `mkdir -p /tmp/hwpx_work && cd /tmp/hwpx_work && unzip -o <path_to_template>`
- List the extracted contents and identify all XML files that might contain `{{...}}` placeholders. Typically these are in `Contents/` directory (e.g., `Contents/section0.xml`, `Contents/section1.xml`, etc.). Check ALL XML files.
- Read each XML file and note all `{{placeholder}}` patterns.

### 2. Write a Python script to perform the replacements
Create a Python script at `/tmp/fill_template.py` that does the following:

```python
import json, os, re, shutil, zipfile

# Paths - adjust if files are in different locations
TEMPLATE = '<path_to_template_hwpx>'  # fill in actual path
JSON_FILE = '<path_to_json>'  # fill in actual path  
OUTPUT = '/root/training_feedback_ready.hwpx'
WORK_DIR = '/tmp/hwpx_edit'

# Clean and extract
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
os.makedirs(WORK_DIR)
with zipfile.ZipFile(TEMPLATE, 'r') as z:
    z.extractall(WORK_DIR)

# Load JSON data
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Process all XML files in the extracted archive
for root, dirs, files in os.walk(WORK_DIR):
    for fname in files:
        if fname.endswith('.xml'):
            fpath = os.path.join(root, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original = content
            
            # Replace placeholders - handle XML tags that may split {{key}}
            # First, create a tag-stripped version to find placeholders,
            # but actually replace in the raw XML
            
            for key, value in data.items():
                val = str(value)
                
                # Special formatting rules:
                if key == '참석자수':
                    # Convert to digits only (e.g., '25명' -> '25')
                    val = re.sub(r'[^0-9]', '', val)
                elif key == '만족도':
                    # Format as 'X.X점 (5.0점 만점)'
                    # Extract numeric score
                    score = re.search(r'[\d.]+', str(value))
                    if score:
                        val = f'{score.group()}점 (5.0점 만점)'
                    else:
                        val = f'{value}점 (5.0점 만점)'
                elif key == '총평':
                    # Append the required sentence
                    val = str(value) + ' 후속 심화반 검토 요망.'
                
                # Build regex that matches {{key}} even with XML tags interspersed
                # The placeholder chars: { { k e y } }
                placeholder_chars = '{{' + key + '}}'
                # Build pattern allowing optional XML tags between each character
                tag_pattern = r'(?:<[^>]*>)*'
                regex_parts = []
                for ch in placeholder_chars:
                    regex_parts.append(re.escape(ch))
                pattern = tag_pattern.join(regex_parts)
                
                content = re.sub(pattern, val, content)
            
            # If content was modified, remove layout cache elements from modified paragraphs
            # Remove all <hp:lineSegArray>...</hp:lineSegArray> from paragraphs that changed
            if content != original:
                # Remove lineSegArray elements (layout cache) to prevent rendering issues
                content = re.sub(r'<hp:lineSegArray>.*?</hp:lineSegArray>', '', content, flags=re.DOTALL)
                
                with open(fpath, 'w', encoding='utf-8') as f:
                    f.write(content)

# Verify no remaining placeholders
remaining = []
for root, dirs, files in os.walk(WORK_DIR):
    for fname in files:
        if fname.endswith('.xml'):
            fpath = os.path.join(root, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read()
            # Check for remaining {{ }} patterns (ignoring XML tags)
            clean = re.sub(r'<[^>]*>', '', text)
            found = re.findall(r'\{\{.*?\}\}', clean)
            if found:
                remaining.extend(found)

if remaining:
    print(f'WARNING: Remaining placeholders: {remaining}')
else:
    print('All placeholders replaced successfully.')

# Re-pack as HWPX (ZIP with stored/deflated entries, preserving structure)
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)

with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zout:
    for root, dirs, files in os.walk(WORK_DIR):
        for fname in files:
            fpath = os.path.join(root, fname)
            arcname = os.path.relpath(fpath, WORK_DIR)
            zout.write(fpath, arcname)

print(f'Output written to {OUTPUT}')
```

### 3. Adjust paths and run
- Before running, verify the actual paths to the template and JSON files.
- Run: `python3 /tmp/fill_template.py`
- Check output for warnings about remaining placeholders.

### 4. Validate the output
- Verify the output file exists: `ls -la /root/training_feedback_ready.hwpx`
- Unzip to a temp location and grep for any remaining `{{` patterns: `mkdir -p /tmp/verify && cd /tmp/verify && unzip -o /root/training_feedback_ready.hwpx && grep -r '{{' . --include='*.xml'`
- Verify specific replacements by grepping for expected values:
  - Check 참석자수 is digits only (no '명' suffix)
  - Check 만족도 has the '점 (5.0점 만점)' format
  - Check 총평 ends with '후속 심화반 검토 요망.'
  - Check no `{{` remains in any XML
- Verify the ZIP structure is intact (has the same directory structure as the original)

### Important Notes
- The HWPX format may split `{{placeholder}}` text across multiple XML elements (e.g., `<hp:t>{{</hp:t><hp:t>key}}</hp:t>`). The regex pattern with optional XML tags between characters handles this.
- Remove `<hp:lineSegArray>` elements from ANY file that was modified, not just specific paragraphs. This is the layout cache that causes rendering artifacts.
- Korean labels and the static note line must remain unchanged — only replace `{{...}}` placeholders.
- The 만족도 value in JSON might be just a number like 4.5 — format it as `4.5점 (5.0점 만점)`.
- The 참석자수 value might be like '25명' or just '25' — extract digits only either way.
- Double-check that the final ZIP at `/root/training_feedback_ready.hwpx` is valid by listing its contents with `unzip -l`.

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