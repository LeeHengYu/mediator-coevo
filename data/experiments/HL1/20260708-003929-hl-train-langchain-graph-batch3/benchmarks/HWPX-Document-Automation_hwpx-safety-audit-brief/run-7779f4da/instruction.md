# Task Instruction

Execute the following steps in order to produce `/root/safety_audit_brief_final.hwpx`.

## 1 – Inspect source files

```bash
cd /root
cat audit_overview.json
cat corrective_actions.json
```

Understand every key/value. Note the inspection date (YYYY-MM-DD format), risk tier, and all overview fields.

## 2 – Inspect the HWPX template

HWPX is a ZIP archive containing XML files.

```bash
python3 -c "
import zipfile, os
with zipfile.ZipFile('safety_audit_template.hwpx') as z:
    for n in z.namelist():
        print(n)
"
```

Then read every XML file inside (especially files under `Contents/` such as `section0.xml` or similar content XML). Print their full text so you can see all `{{...}}` placeholders, section titles, row labels, and the document structure.

```bash
python3 -c "
import zipfile
with zipfile.ZipFile('safety_audit_template.hwpx') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            print('===', name, '===')
            print(z.read(name).decode('utf-8', errors='replace'))
"
```

## 3 – Build the output document

Write a Python script (`build.py`) that:

### 3a – Load data
- Reads `audit_overview.json` and `corrective_actions.json`.

### 3b – Prepare substitutions
- Build a dictionary mapping every `{{placeholder}}` to its replacement value.
- For the inspection date: convert from `YYYY-MM-DD` to `YYYY.MM.DD` format. Every occurrence of the date in the document must use the dot-separated form.
- For the risk tier: after substituting the risk tier value (e.g., "High"), also append a severity note using this mapping:
  - High → 즉시조치
  - Medium → 계획보완  
  - Low → 모니터링
  
  The severity note should appear immediately after the risk tier text, separated by a space (e.g., "High 즉시조치"). Apply this everywhere the risk tier appears.
- For corrective actions: fill the three corrective-action lines in the same order they appear in `corrective_actions.json`.

### 3c – Process XML files inside the ZIP

Because XML tags can split placeholder text across multiple text nodes, use **string-level replacement on the raw XML text** (not DOM text-node-only replacement) for each placeholder. This is the proven approach from prior HWPX tasks.

Specifically:
1. Read each file from the ZIP as a UTF-8 string.
2. For each `{{placeholder}}`, do `xml_text = xml_text.replace('{{placeholder}}', value)`.
3. After all replacements, verify no `{{` remains in any XML file. If any remain, print them and abort.

### 3d – Clear layout caches

After placeholder replacement, parse each content XML with lxml and remove all elements whose local name is `lineSegArray` (these are stale layout caches). Use:

```python
from lxml import etree
tree = etree.fromstring(xml_bytes)
for elem in tree.xpath('//*[local-name()="lineSegArray"]'):
    elem.getparent().remove(elem)
xml_bytes = etree.tostring(tree, xml_declaration=True, encoding='UTF-8')
```

### 3e – Rebuild the ZIP

- Write the output to `/root/safety_audit_brief_final.hwpx`.
- If the original ZIP contains a `mimetype` entry, write it first with `ZIP_STORED` (no compression) to preserve ODF/HWPX package conventions.
- Write all other entries with `ZIP_DEFLATED`.
- Preserve the original entry list; do not add or remove files.

## 4 – Validate

```bash
python3 -c "
import zipfile
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            if '{{' in content:
                print('LEFTOVER PLACEHOLDER in', name)
                # print surrounding context
                idx = content.index('{{')
                print(content[max(0,idx-80):idx+80])
            else:
                print(name, 'OK')
    print('All entries:', z.namelist())
"
```

Also verify the date format and risk tier + severity note appear correctly:

```bash
python3 -c "
import zipfile, re
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            c = z.read(name).decode('utf-8')
            # Check date format
            dates = re.findall(r'\d{4}[.-]\d{2}[.-]\d{2}', c)
            for d in dates:
                print(name, 'date:', d)
            # Check risk tier + severity
            for kw in ['즉시조치','계획보완','모니터링']:
                if kw in c:
                    idx = c.index(kw)
                    print(name, 'severity context:', c[max(0,idx-30):idx+30])
"
```

If any check fails, diagnose and fix before finishing.

## 5 – Run the verifier if available

```bash
cd /root && ls test_output.py 2>/dev/null && python3 -m pytest test_output.py -v
```

If tests fail, read the error messages carefully, fix the output, and re-run until all tests pass.

## Key Reminders
- Do NOT change section titles or row labels.
- Do NOT leave any `{{...}}` placeholders.
- The risk tier AND severity note must appear everywhere the risk tier is referenced.
- The date must be `YYYY.MM.DD` everywhere (no hyphens).
- Corrective actions must be in the same order as `corrective_actions.json`.
- Remove `lineSegArray` elements from every modified paragraph to prevent overlapping characters.
- The result must be a valid ZIP (hwpx package).

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