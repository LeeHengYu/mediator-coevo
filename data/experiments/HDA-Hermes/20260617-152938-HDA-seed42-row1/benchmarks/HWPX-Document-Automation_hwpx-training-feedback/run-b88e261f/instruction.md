# Task Instruction

Execute the following steps to fill in the training feedback HWPX template and produce `/root/training_feedback_ready.hwpx`:

1. **Inspect the workspace.** List files in the task directory to locate `training_feedback_template.hwpx` and `training_feedback.json`. Also check for any `test_output.py` or verifier script so you understand the acceptance criteria.

2. **Read the JSON data file** (`training_feedback.json`) and note every key-value pair. Pay special attention to:
   - `참석자수` – you will strip all non-digit characters before inserting.
   - `만족도` – you will reformat as `X.X점 (5.0점 만점)` using the numeric score from JSON.
   - The overall-opinion field – you will append ` 후속 심화반 검토 요망.` after the provided comment.

3. **Unzip the template.** Extract `training_feedback_template.hwpx` into a temporary directory (e.g., `/tmp/hwpx_work/`). List the extracted contents to understand the package structure.

4. **Identify XML files containing placeholders.** Search all extracted XML files (especially under `Contents/`) for `{{` to find every placeholder. Record the file paths and the exact placeholder strings.

5. **Write a Python script** that does the following:
   ```python
   import json, os, re, shutil, zipfile
   from lxml import etree

   # Paths
   JSON_PATH = '<path to training_feedback.json>'
   EXTRACT_DIR = '/tmp/hwpx_work'
   OUTPUT_PATH = '/root/training_feedback_ready.hwpx'

   # Load JSON
   with open(JSON_PATH) as f:
       data = json.load(f)

   # Build replacement map from JSON keys to final display values
   replacements = {}
   for key, value in data.items():
       str_val = str(value)
       if key == '참석자수':
           str_val = re.sub(r'[^0-9]', '', str_val)
       elif key == '만족도':
           # Extract numeric score, format as "X.X점 (5.0점 만점)"
           score = str(value)  # e.g. "4.5" or 4.5
           str_val = f'{score}점 (5.0점 만점)'
       replacements[key] = str_val

   # Find the overall-opinion key (inspect JSON to determine exact key name)
   # Append the required suffix to the overall opinion value
   # The key might be '종합의견' or similar – adapt after inspecting JSON
   for key in data:
       if '종합' in key or '의견' in key or 'opinion' in key.lower():
           original = replacements[key]
           if not original.endswith(' 후속 심화반 검토 요망.'):
               replacements[key] = original.rstrip() + ' 후속 심화반 검토 요망.'

   # Process each XML file in the extracted HWPX
   for root_dir, dirs, files in os.walk(EXTRACT_DIR):
       for fname in files:
           fpath = os.path.join(root_dir, fname)
           if not fname.endswith('.xml'):
               continue
           with open(fpath, 'rb') as f:
               raw = f.read()
           text = raw.decode('utf-8')
           if '{{' not in text:
               continue

           # Parse with lxml to handle namespace-aware operations
           tree = etree.fromstring(raw)

           # Replace placeholders in all text nodes
           for elem in tree.iter():
               if elem.text and '{{' in elem.text:
                   for key, val in replacements.items():
                       elem.text = elem.text.replace('{{' + key + '}}', val)
               if elem.tail and '{{' in elem.tail:
                   for key, val in replacements.items():
                       elem.tail = elem.tail.replace('{{' + key + '}}', val)

           # Remove lineSegArray elements (layout cache) from paragraphs
           # that were modified. To be safe, remove ALL lineSegArray elements.
           nsmap = tree.nsmap
           hp_ns = None
           for prefix, uri in nsmap.items():
               if 'hwpml' in uri or prefix == 'hp':
                   hp_ns = uri
                   break
           # Also try common namespace patterns
           for ns_uri in [hp_ns, 'http://www.hancom.co.kr/hwpml/2011/paragraph',
                          'http://www.hancom.co.kr/hwpml/2011/head',
                          'urn:hancom:hwpml:2011']:
               if ns_uri:
                   for lsa in tree.findall(f'.//{{{ns_uri}}}lineSegArray'):
                       lsa.getparent().remove(lsa)
           # Also try without namespace
           for lsa in tree.iter():
               local = etree.QName(lsa.tag).localname if isinstance(lsa.tag, str) else ''
               if local == 'lineSegArray':
                   lsa.getparent().remove(lsa)

           # Write back
           result = etree.tostring(tree, xml_declaration=True, encoding='UTF-8')
           with open(fpath, 'wb') as f:
               f.write(result)

   # Verify no {{...}} remain
   for root_dir, dirs, files in os.walk(EXTRACT_DIR):
       for fname in files:
           fpath = os.path.join(root_dir, fname)
           with open(fpath, 'rb') as f:
               content = f.read()
           if b'{{' in content:
               print(f'WARNING: Residual placeholder in {fpath}')
               print(content[content.index(b'{{'):content.index(b'}}')+2])

   # Repackage as HWPX (ZIP)
   # Preserve original file order and compression from the template
   TEMPLATE_PATH = '<path to training_feedback_template.hwpx>'
   with zipfile.ZipFile(TEMPLATE_PATH, 'r') as orig_zip:
       orig_names = orig_zip.namelist()

   with zipfile.ZipFile(OUTPUT_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
       for name in orig_names:
           file_on_disk = os.path.join(EXTRACT_DIR, name)
           if os.path.isfile(file_on_disk):
               # mimetype should be stored, not deflated (if present)
               compress = zipfile.ZIP_STORED if name == 'mimetype' else zipfile.ZIP_DEFLATED
               zf.write(file_on_disk, name, compress_type=compress)

   print(f'Output written to {OUTPUT_PATH}')
   ```

6. **Adapt the script after inspection.** Before running, adjust:
   - The exact JSON key for the overall opinion field (inspect the JSON first).
   - The exact paths to template and JSON files.
   - If the overall opinion placeholder pattern differs (e.g., the placeholder might be inside a longer sentence), handle accordingly.

7. **Run the script** and check for any warnings about residual placeholders.

8. **Validate the output:**
   - Confirm `/root/training_feedback_ready.hwpx` exists and is a valid ZIP: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/training_feedback_ready.hwpx'); print(z.namelist())"`
   - Search for any remaining `{{` in the output: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/training_feedback_ready.hwpx'); [print(n) for n in z.namelist() if b'{{' in z.read(n)]"`
   - Verify key values appear: check that the digits-only attendee count, the `점 (5.0점 만점)` satisfaction format, and `후속 심화반 검토 요망.` all appear in the XML content.
   - Verify no `lineSegArray` elements remain in modified paragraphs.

9. **Run the verifier** if a test script exists: `cd <task_dir> && python -m pytest test_output.py -v`

Key cautions:
- Do NOT assume placeholder key names; read the JSON first and match exactly.
- The overall-opinion suffix must be appended with a space separator, ending with a period.
- `lineSegArray` removal must cover all namespaces used in the document; iterate by local name as a fallback.
- Preserve the original ZIP entry ordering and ensure no extra directory entries are added.
- Korean labels and the static note line must remain untouched.

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