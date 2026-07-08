# Task Instruction

You must produce `/root/safety_audit_brief_final.hwpx` by filling the template with data from the two JSON files.

## Step 0 – Inspect all inputs

```bash
cd /root
ls -la
cat audit_overview.json
cat corrective_actions.json
```

Then inspect the HWPX template (it is a ZIP):

```bash
python3 -c "
import zipfile, os
with zipfile.ZipFile('safety_audit_template.hwpx') as z:
    z.printdir()
"
```

Extract and print the XML section files (section0.xml, section1.xml, and any others under Contents/):

```bash
python3 -c "
import zipfile
with zipfile.ZipFile('safety_audit_template.hwpx') as z:
    for name in z.namelist():
        if name.endswith('.xml') or name.endswith('.rels'):
            print('=== ' + name + ' ===')
            print(z.read(name).decode('utf-8', errors='replace'))
"
```

Also look at the test/verifier script to understand exactly what is checked:

```bash
find /root -name '*.py' -path '*test*' | head -20
find /root -name 'verify*' -o -name 'check*' -o -name 'grade*' | head -20
```

Read whatever verifier/test file you find so you know the exact assertions.

## Step 1 – Understand the verifier contract

From previous feedback, the verifier does:
```python
assert f'{value} ({severity_map[value]})' in combined
```
where `severity_map = {'High': '즉시조치', 'Medium': '계획보완', 'Low': '모니터링'}`.

So the risk tier must appear as **`High (즉시조치)`** (with parentheses and a space before the opening paren). Confirm this by reading the actual test file.

Also check what other assertions exist (date format `YYYY.MM.DD`, no remaining `{{…}}` placeholders, corrective action order, valid ZIP, etc.).

## Step 2 – Write the build script

Create `/root/build.py` that:

1. Loads `audit_overview.json` and `corrective_actions.json`.
2. Opens `safety_audit_template.hwpx` as a ZIP.
3. For every file in the ZIP:
   a. If it is an XML section file (e.g., `Contents/section0.xml`, `Contents/section1.xml`):
      - Read as UTF-8 text.
      - Replace all `{{placeholder}}` tokens with the corresponding JSON values.
        * Map each placeholder to the right JSON field. Inspect the template XML carefully to see exact placeholder names.
      - For the risk tier placeholder: replace `{{risk_tier}}` (or whatever it is) with the value from JSON **plus** the severity note in parentheses, e.g., `High (즉시조치)`. Use the mapping: High→즉시조치, Medium→계획보완, Low→모니터링.
      - **Every occurrence** of the risk tier value in the document (even if it was already substituted or appears in other placeholders) must have the severity note appended. After initial placeholder substitution, do a second pass: for each tier in [High, Medium, Low], replace any bare occurrence of that tier word that doesn't already have the note.
      - Rewrite dates from `YYYY-MM-DD` to `YYYY.MM.DD` everywhere (use regex `r'(\d{4})-(\d{2})-(\d{2})'` → `r'\1.\2.\3'`).
      - Fill the three corrective-action lines in the order they appear in `corrective_actions.json`.
      - Remove all `lineSegArray` elements (layout cache) from any paragraph whose text was modified. Use lxml to parse, find all `lineSegArray` elements (using local-name() to handle namespaces), remove them, then serialize back. **Important**: do placeholder substitution as string operations first, then parse with lxml to remove lineSegArray, then serialize back.
      - Verify no `{{` remains in the output text.
   b. Otherwise, copy the file unchanged.
4. Write the result to `/root/safety_audit_brief_final.hwpx` as a valid ZIP, preserving the `mimetype` entry first (uncompressed, if it was uncompressed in the original) and all other entries.

### Key details for the build script:

- Use `import zipfile, json, re, copy` and `from lxml import etree`.
- When rebuilding the ZIP, iterate over the original ZipFile's `infolist()` to preserve compression types. Write `mimetype` first if it exists, with `compress_type=zipfile.ZIP_STORED`.
- For lineSegArray removal: after string substitutions, parse with `etree.fromstring(xml_bytes)`, find all elements where `local-name()='lineSegArray'`, remove each from its parent, then `etree.tostring(root, xml_declaration=True, encoding='utf-8')`.
- Actually, be more targeted: only remove lineSegArray from paragraphs that contain modified text. But for safety and simplicity, removing ALL lineSegArray elements is acceptable (the cross-task artifact confirms this approach works).

## Step 3 – Run the build

```bash
cd /root
python3 build.py
```

If errors occur, fix them and re-run.

## Step 4 – Validate the output

1. Confirm the output is a valid ZIP:
```bash
python3 -c "import zipfile; z=zipfile.ZipFile('/root/safety_audit_brief_final.hwpx'); z.testzip(); print('Valid ZIP'); z.printdir()"
```

2. Extract and print section XMLs from the output to manually verify:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx') as z:
    for name in z.namelist():
        if 'section' in name and name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            print('=== ' + name + ' ===')
            print(content[:5000])
"
```

3. Check no `{{` placeholders remain:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            if '{{' in content:
                print(f'FAIL: placeholder found in {name}')
                import re
                for m in re.finditer(r'\{\{.*?\}\}', content):
                    print(f'  {m.group()}')
            else:
                print(f'OK: {name}')
"
```

4. Check risk tier format:
```bash
python3 -c "
import zipfile, json
severity_map = {'High': '즉시조치', 'Medium': '계획보완', 'Low': '모니터링'}
with open('audit_overview.json') as f:
    overview = json.load(f)
risk = overview.get('risk_tier') or overview.get('risk_level') or overview.get('위험등급', '')
print(f'Risk tier from JSON: {risk}')
expected = f'{risk} ({severity_map.get(risk, "?")})'  
print(f'Expected string: {expected}')
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            if expected in content:
                print(f'FOUND in {name}')
"
```

5. Check date format (should be YYYY.MM.DD, no YYYY-MM-DD):
```bash
python3 -c "
import zipfile, re
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            dashes = re.findall(r'\d{4}-\d{2}-\d{2}', content)
            dots = re.findall(r'\d{4}\.\d{2}\.\d{2}', content)
            if dashes:
                print(f'FAIL: dash dates in {name}: {dashes}')
            if dots:
                print(f'OK: dot dates in {name}: {dots}')
"
```

6. Run the actual verifier/test if you found one in Step 0:
```bash
# Replace with actual test command found earlier
python3 -m pytest /root/tests/ -v 2>&1 | tail -40
```

Fix any failures and iterate until all checks pass.

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