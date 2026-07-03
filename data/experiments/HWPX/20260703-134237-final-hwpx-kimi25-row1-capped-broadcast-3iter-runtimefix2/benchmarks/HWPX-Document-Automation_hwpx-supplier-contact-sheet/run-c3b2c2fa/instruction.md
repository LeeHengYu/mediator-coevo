# Task Instruction

You need to update the HWPX supplier contact sheet template with values from a JSON file. Follow these steps precisely:

1. **Inspect the workspace:**
   ```bash
   ls /root/
   find /root/ -name 'supplier_contact_template.hwpx' -o -name 'supplier_contact.json' 2>/dev/null
   ```
   Also check the current working directory:
   ```bash
   ls -la
   find . -name 'supplier_contact_template.hwpx' -o -name 'supplier_contact.json'
   ```

2. **Read the JSON data:**
   ```bash
   cat supplier_contact.json
   ```
   (adjust path as needed based on step 1)

3. **Examine the HWPX template structure:**
   The `.hwpx` file is a ZIP archive. List its contents:
   ```bash
   python3 -c "import zipfile; z=zipfile.ZipFile('supplier_contact_template.hwpx'); print('\n'.join(z.namelist()))"
   ```

4. **Inspect all XML files inside the HWPX** to find `{{...}}` placeholders:
   ```bash
   python3 << 'PYEOF'
import zipfile
z = zipfile.ZipFile('supplier_contact_template.hwpx')
for name in z.namelist():
    data = z.read(name)
    try:
        text = data.decode('utf-8')
        if '{{' in text:
            print(f"=== {name} (contains placeholders) ===")
            print(text[:5000])
            print("...")
    except:
        pass
PYEOF
   ```
   Read the full content of any XML files containing placeholders (especially section0.xml and any others). Make sure you see ALL placeholders.

5. **Write a Python script to perform the replacement:**

   Key requirements for the script:
   - Load the JSON file. If it contains nested objects/arrays, flatten them so that `{{key.subkey}}` or `{{key}}` patterns can be matched. Also handle the case where the JSON might use simple top-level keys.
   - Open the HWPX ZIP, iterate through all entries.
   - For each XML/text entry, decode to string, perform replacements of all `{{placeholder}}` patterns with corresponding JSON values.
   - **Critical:** After replacing text in any paragraph, remove all `<hp:lineSegArray>` elements (and their contents) from the modified XML. This clears stale layout cache and prevents overlapping characters when the document is opened. Use regex: `re.sub(r'<hp:lineSegArray>.*?</hp:lineSegArray>', '', text, flags=re.DOTALL)`
   - Write the result to `/root/supplier_contact_ready.hwpx` as a new ZIP, preserving all original entries (binary files copied as-is, text/XML files with replacements applied).
   - After writing, verify: re-open the output HWPX and scan ALL entries for any remaining `{{` patterns. Print any found. Assert none remain.

   Here is a template script (adapt based on what you find in steps 2-4):
   ```python
   import zipfile, json, re, os, shutil

   # Load JSON
   with open('supplier_contact.json', 'r', encoding='utf-8') as f:
       data = json.load(f)

   # Flatten nested JSON
   def flatten(obj, prefix=''):
       items = {}
       if isinstance(obj, dict):
           for k, v in obj.items():
               new_key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
               if isinstance(v, (dict, list)):
                   items.update(flatten(v, new_key))
               else:
                   items[new_key] = str(v)
       elif isinstance(obj, list):
           for i, v in enumerate(obj):
               new_key = f"{prefix}[{i}]"
               if isinstance(v, (dict, list)):
                   items.update(flatten(v, new_key))
               else:
                   items[new_key] = str(v)
       return items

   flat = flatten(data)
   # Also keep top-level keys as-is
   if isinstance(data, dict):
       for k, v in data.items():
           if not isinstance(v, (dict, list)):
               flat[k] = str(v)

   print("Replacement map:")
   for k, v in flat.items():
       print(f"  {{{{{k}}}}} -> {v}")

   # Process HWPX
   src = 'supplier_contact_template.hwpx'
   dst = '/root/supplier_contact_ready.hwpx'

   with zipfile.ZipFile(src, 'r') as zin, zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
       for item in zin.infolist():
           raw = zin.read(item.filename)
           try:
               text = raw.decode('utf-8')
               modified = False
               for key, val in flat.items():
                   placeholder = '{{' + key + '}}'
                   if placeholder in text:
                       text = text.replace(placeholder, val)
                       modified = True
               # Remove stale layout cache from modified paragraphs
               if modified:
                   text = re.sub(r'<hp:lineSegArray>.*?</hp:lineSegArray>', '', text, flags=re.DOTALL)
               zout.writestr(item, text)
           except (UnicodeDecodeError, Exception):
               zout.writestr(item, raw)

   # Verify no placeholders remain
   print("\nVerification:")
   remaining = []
   with zipfile.ZipFile(dst, 'r') as z:
       for name in z.namelist():
           try:
               content = z.read(name).decode('utf-8')
               found = re.findall(r'\{\{[^}]+\}\}', content)
               if found:
                   remaining.extend([(name, p) for p in found])
                   print(f"  WARNING: {name} still has: {found}")
           except:
               pass
   if remaining:
       print(f"FAIL: {len(remaining)} placeholders remain!")
   else:
       print("SUCCESS: No placeholders remain.")
   print(f"Output written to {dst}")
   ```

6. **Run the script and check output:**
   - Ensure zero remaining placeholders.
   - Verify the output file exists and is a valid ZIP:
     ```bash
     python3 -c "import zipfile; z=zipfile.ZipFile('/root/supplier_contact_ready.hwpx'); print('Valid ZIP with', len(z.namelist()), 'entries')"
     ```

7. **If any test file exists**, run it:
   ```bash
   find . -name 'test_output*' -o -name 'test_*' | head -5
   # If found:
   cd /path/to/tests && python3 -m pytest -xvs
   ```

**Important edge cases to watch for:**
- The JSON may have nested structures (e.g., contact details within supplier objects). Inspect the actual placeholder names in the XML and the actual JSON keys carefully. If the placeholders use dot notation like `{{supplier.name}}`, ensure your flattening produces matching keys.
- If the JSON has array items that map to placeholders like `{{items[0].name}}`, handle those too.
- Some placeholders might span across XML tags (e.g., `<hp:t>{{</hp:t><hp:t>name}}</hp:t>`). Check for this pattern. If found, you may need to first consolidate adjacent `<hp:t>` elements or do a broader regex replacement.
- Preserve all Korean labels and static note lines — only replace `{{...}}` patterns.
- Make sure the lineSegArray removal only happens on files that were actually modified, to avoid unnecessary changes to unmodified XML files.

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