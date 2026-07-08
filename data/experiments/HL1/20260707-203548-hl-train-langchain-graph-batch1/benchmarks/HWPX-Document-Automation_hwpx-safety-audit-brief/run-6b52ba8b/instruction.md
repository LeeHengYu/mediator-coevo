# Task Instruction

Complete the following task to produce `/root/safety_audit_brief_final.hwpx` from a template and two JSON data files.

## Context

HWPX files are ZIP archives containing XML content files (typically under `Contents/` with files like `section0.xml`). Placeholders like `{{key}}` may be split across multiple XML `<hp:t>` tags (e.g., `<hp:t>{{</hp:t><hp:t>key}}</hp:t>`). Layout cache elements (`lineSegArray`, `lineSeg`) in paragraphs whose text is modified must be removed to prevent overlapping characters.

## Steps

### 1. Explore the workspace
```bash
find /root -maxdepth 3 -type f | head -60
```
Identify the template file `safety_audit_template.hwpx`, `audit_overview.json`, and `corrective_actions.json`. Also look for any test/verifier files.

### 2. Read the JSON data files
```bash
cat <path_to_audit_overview.json>
cat <path_to_corrective_actions.json>
```
Note all field names, values, the risk tier value, and the inspection date (in YYYY-MM-DD format).

### 3. Inspect the HWPX template structure
```bash
python3 -c "
import zipfile, sys
z = zipfile.ZipFile('<path_to_template>')
for n in z.namelist():
    print(n)
"
```
Then extract and read the main content XML (likely `Contents/section0.xml`) to see all placeholders and the document structure:
```bash
python3 -c "
import zipfile
z = zipfile.ZipFile('<path_to_template>')
for name in z.namelist():
    if 'section' in name.lower() and name.endswith('.xml'):
        print('=== ' + name + ' ===')
        print(z.read(name).decode('utf-8'))
"
```

### 4. Identify all placeholders
List every `{{...}}` placeholder in the XML content, including ones that might be split across tags. Map each placeholder to the corresponding JSON field.

### 5. Write and run the Python transformation script

Create a Python script that does the following:

```python
import zipfile, json, re, os, shutil, io

# Paths - adjust based on what you found in step 1
TEMPLATE = '<path_to_template>'
OVERVIEW_JSON = '<path_to_audit_overview.json>'
ACTIONS_JSON = '<path_to_corrective_actions.json>'
OUTPUT = '/root/safety_audit_brief_final.hwpx'

# Load JSON data
with open(OVERVIEW_JSON) as f:
    overview = json.load(f)
with open(ACTIONS_JSON) as f:
    actions = json.load(f)

# Build replacement map from both JSON files
# Include all overview fields and corrective action fields
# For corrective actions, map placeholders like {{action_1_field}} to values
# in the order they appear in the JSON array

# Severity mapping
severity_map = {'High': '즉시조치', 'Medium': '계획보완', 'Low': '모니터링'}

# Date reformatting: YYYY-MM-DD -> YYYY.MM.DD
# Risk tier: append severity note

# Process the HWPX
with zipfile.ZipFile(TEMPLATE, 'r') as zin:
    with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.namelist():
            data = zin.read(item)
            if item.endswith('.xml'):
                text = data.decode('utf-8')
                
                # Step A: Normalize split placeholders
                # Remove XML tags between {{ and }} so split placeholders become whole
                # Pattern: find {{ ... }} where ... may contain </hp:t><hp:t> etc.
                def merge_split_placeholders(xml_text):
                    # Repeatedly merge tags inside placeholder patterns
                    pattern = r'(\{\{[^}]*?)<\/hp:t>\s*<hp:t[^>]*>([^}]*?\}\})'
                    prev = None
                    while prev != xml_text:
                        prev = xml_text
                        xml_text = re.sub(pattern, r'\1\2', xml_text)
                    # Also handle partial: opening braces in one tag, rest in next
                    pattern2 = r'(\{\{[^}]*?)<\/hp:t>\s*<hp:t[^>]*>([^}]*?)'
                    # Be more careful - only merge if we can see this is part of a placeholder
                    prev = None
                    while prev != xml_text:
                        prev = xml_text
                        xml_text = re.sub(r'(\{\{(?:(?!\}\}).)*?)<\/hp:t>\s*<hp:t[^>]*>((?:(?!\{\{).)*?\}\})', r'\1\2', xml_text)
                    return xml_text
                
                text = merge_split_placeholders(text)
                
                # Step B: Replace all {{placeholder}} with values from the replacement map
                # Build the map by inspecting what placeholders exist
                # Replace each one
                
                # Step C: Reformat dates YYYY-MM-DD -> YYYY.MM.DD everywhere
                text = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\1.\2.\3', text)
                
                # Step D: After risk tier value, append severity note
                # e.g., if risk_tier is "High", replace "High" with "High 즉시조치"
                # But only for the risk tier occurrences
                
                # Step E: Remove layout cache elements from modified paragraphs
                # Remove all <hp:lineSegArray>...</hp:lineSegArray> elements
                text = re.sub(r'<hp:lineSegArray[^>]*>.*?</hp:lineSegArray>', '', text, flags=re.DOTALL)
                # Also remove standalone <hp:lineSeg.../> or <hp:lineSeg>...</hp:lineSeg>
                text = re.sub(r'<hp:lineSeg[^/]*?/>', '', text)
                text = re.sub(r'<hp:lineSeg[^>]*>.*?</hp:lineSeg>', '', text, flags=re.DOTALL)
                
                # Step F: Verify no {{...}} placeholders remain
                remaining = re.findall(r'\{\{.*?\}\}', text)
                if remaining:
                    print(f'WARNING: Unreplaced placeholders in {item}: {remaining}')
                
                data = text.encode('utf-8')
            zout.writestr(item, data)

print('Output written to', OUTPUT)
```

**IMPORTANT NOTES for the script:**
- You MUST first inspect the actual placeholder names in the template XML and the actual JSON field names before writing the final script. The placeholder names and JSON keys must match exactly.
- For corrective actions, determine the exact placeholder naming convention from the template (e.g., `{{corrective_action_1_description}}` or `{{action1_desc}}` etc.) and map them to the JSON array entries in order.
- The risk tier value comes from the JSON. After replacing the placeholder with the value, also find every other occurrence of that risk tier text and append the severity note. Use the severity_map to determine the Korean text.
- The severity note should be appended with a space: e.g., `"High 즉시조치"`.
- Make sure the date reformatting happens AFTER placeholder replacement so the dates from JSON get reformatted too.
- The order of operations should be: (1) merge split placeholders, (2) replace all placeholders with JSON values, (3) reformat dates, (4) append severity notes to risk tier, (5) strip layout caches.

### 6. Validate the output
```bash
# Check it's a valid ZIP/HWPX
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/safety_audit_brief_final.hwpx')
for n in z.namelist():
    print(n)
"

# Check no remaining placeholders
python3 -c "
import zipfile, re
z = zipfile.ZipFile('/root/safety_audit_brief_final.hwpx')
for name in z.namelist():
    if name.endswith('.xml'):
        text = z.read(name).decode('utf-8')
        found = re.findall(r'\{\{.*?\}\}', text)
        if found:
            print(f'FAIL: {name} has placeholders: {found}')
        else:
            print(f'OK: {name}')
"

# Check dates are in YYYY.MM.DD format (no YYYY-MM-DD)
python3 -c "
import zipfile, re
z = zipfile.ZipFile('/root/safety_audit_brief_final.hwpx')
for name in z.namelist():
    if name.endswith('.xml'):
        text = z.read(name).decode('utf-8')
        bad_dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
        if bad_dates:
            print(f'FAIL: {name} has old date format: {bad_dates}')
"

# Check severity note is present
# Check no lineSegArray or lineSeg elements remain
python3 -c "
import zipfile, re
z = zipfile.ZipFile('/root/safety_audit_brief_final.hwpx')
for name in z.namelist():
    if name.endswith('.xml'):
        text = z.read(name).decode('utf-8')
        if 'lineSegArray' in text or '<hp:lineSeg' in text:
            print(f'FAIL: {name} still has layout cache elements')
        else:
            print(f'OK layout cache: {name}')
"
```

### 7. Run the verifier if available
```bash
cd /root && find . -name 'test_*.py' -o -name 'verify*.py' | head -5
# Then run: pytest <test_file> -v
```

## Critical Reminders
- Do NOT write the transformation script until you have inspected the actual template XML and both JSON files. The placeholder names, JSON structure, and corrective action format must be determined from the actual files.
- Corrective actions must be filled in the SAME ORDER as they appear in `corrective_actions.json`.
- EVERY occurrence of the risk tier must be updated (not just the first one).
- The severity note goes IMMEDIATELY AFTER the risk tier text (with a space), everywhere the risk tier appears.
- Section titles and row labels must be preserved exactly.
- The output must be at exactly `/root/safety_audit_brief_final.hwpx`.

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