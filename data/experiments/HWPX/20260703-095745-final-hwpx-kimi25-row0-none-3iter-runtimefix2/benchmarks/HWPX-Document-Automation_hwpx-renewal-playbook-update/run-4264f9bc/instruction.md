# Task Instruction

Execute the following steps to produce `/root/renewal_playbook_updated.hwpx`.

## 0 — Orientation

```bash
cd /root
ls -la renewal_playbook.hwpx renewal_update.json followups.csv
cat renewal_update.json
cat followups.csv
```

Understand the update payload (field names, old→new values) and the follow-up rows (sequence-ordered lines).

## 1 — Unpack the HWPX

HWPX is a ZIP archive containing XML files.

```bash
mkdir -p /root/hwpx_work
cd /root/hwpx_work
unzip -o /root/renewal_playbook.hwpx -d original
cp -a original modified
```

## 2 — Inspect the XML sections

```bash
find /root/hwpx_work/original -name '*.xml' | sort
# Focus on section XMLs (usually Contents/section0.xml, section1.xml, etc.)
for f in /root/hwpx_work/original/Contents/section*.xml; do
  echo "=== $f ==="
  cat "$f"
  echo
done
```

Read the full raw XML carefully. Identify:
- Every place the old customer name, owner, renewal window, pricing band, escalation contact, and pricing note appear.
- The three existing follow-up lines.
- The appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` (must stay unchanged).
- How text is distributed across `<hp:run>` / `<hp:t>` elements — text may be split across multiple runs.

## 3 — Write a Python replacement script

Create `/root/hwpx_work/apply_updates.py` with the following logic:

### 3a — Load update data
```python
import json, csv, os, re, shutil
from lxml import etree

with open('/root/renewal_update.json') as f:
    updates = json.load(f)

with open('/root/followups.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    followup_rows = sorted(reader, key=lambda r: int(r['sequence']))
```

### 3b — Build old→new replacement map
From `renewal_update.json`, build a dict mapping each old value to its new value for every field (customer_name, current_owner, renewal_window, pricing_band, escalation_contact, pricing_note).

### 3c — Process each section XML
For each `section*.xml` under `modified/Contents/`:

1. **Parse the XML** with `lxml.etree` preserving namespaces.
2. **Serialize the full XML to a UTF-8 string.**
3. **Concatenate-and-replace strategy for split tags:**
   - For each `<hp:p>` paragraph element, collect ALL descendant `<hp:t>` elements in document order.
   - Concatenate their `.text` values to form the full paragraph text.
   - For each old→new pair in the replacement map, check if the old value appears in the concatenated text.
   - If it does, perform the replacement by rewriting the `<hp:t>` nodes: put the entire replaced paragraph text into the FIRST `<hp:t>` node and clear (set to empty string) all subsequent `<hp:t>` nodes in that paragraph. This avoids split-tag problems.
   - Track whether any modification was made to this paragraph.
4. **Follow-up line replacement:**
   - Identify the three existing follow-up paragraphs. They likely contain recognizable patterns (e.g., numbered items, or specific text from the original). Collect them.
   - Replace them with the CSV follow-up items in `sequence` order. Put each follow-up item's text (e.g., the relevant column like `action` or `description` — inspect the CSV headers to determine) into the corresponding paragraph's first `<hp:t>`, clearing the rest.
   - If there are more or fewer CSV rows than existing follow-up paragraphs, add or remove `<hp:p>` elements as needed (clone an existing follow-up paragraph as template for additions; remove extras).
5. **Remove layout-cache from modified paragraphs:**
   - For every `<hp:p>` that was modified, find and remove ALL `<hp:lineSegArray>` (or `<hp:linesegarray>` — check namespace) child elements. This is CRITICAL to prevent overlapping characters.
6. **Verify the appendix sentence** `이 부록 문단은 그대로 유지해야 합니다.` is still present and unmodified.
7. **Write the modified XML** back to the same path under `modified/`.

### 3d — Repackage the HWPX
```python
import zipfile

output_path = '/root/renewal_playbook_updated.hwpx'
original_zip = '/root/renewal_playbook.hwpx'
modified_dir = '/root/hwpx_work/modified'

# Preserve the original ZIP's member order and compression
with zipfile.ZipFile(original_zip, 'r') as zin:
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            filepath = os.path.join(modified_dir, item.filename)
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as fmod:
                    zout.writestr(item, fmod.read())
            elif not item.is_dir():
                zout.writestr(item, zin.read(item.filename))
```

Run the script:
```bash
cd /root/hwpx_work
python3 apply_updates.py
```

## 4 — Validate the output

```bash
# 4a — Check it's a valid ZIP
python3 -c "import zipfile; z=zipfile.ZipFile('/root/renewal_playbook_updated.hwpx'); z.testzip(); print('ZIP OK')"

# 4b — Extract and verify content
mkdir -p /root/hwpx_verify
unzip -o /root/renewal_playbook_updated.hwpx -d /root/hwpx_verify

# 4c — Check all new values appear in the section XMLs
python3 << 'PYEOF'
import json
with open('/root/renewal_update.json') as f:
    updates = json.load(f)

sections = []
import glob
for p in sorted(glob.glob('/root/hwpx_verify/Contents/section*.xml')):
    with open(p, encoding='utf-8') as f:
        sections.append(f.read())
combined = '\n'.join(sections)

for field, mapping in updates.items():
    new_val = mapping.get('new') or mapping.get('to') or (mapping if isinstance(mapping, str) else None)
    if new_val is None:
        # Try other structures
        continue
    assert new_val in combined, f'MISSING new value for {field}: {new_val!r}'
    old_val = mapping.get('old') or mapping.get('from')
    if old_val:
        assert old_val not in combined, f'STALE old value for {field}: {old_val!r}'

print('All update values verified.')

# Check appendix
assert '이 부록 문단은 그대로 유지해야 합니다.' in combined, 'Appendix sentence missing!'
print('Appendix OK.')

# Check follow-ups present
import csv
with open('/root/followups.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in sorted(reader, key=lambda r: int(r['sequence'])):
        # Check some identifying text from each row appears
        for col in row:
            if col != 'sequence' and row[col].strip():
                assert row[col].strip() in combined, f'Follow-up text missing: {row[col]!r}'
                break
print('Follow-ups OK.')
PYEOF

# 4d — Check no lineSegArray in modified paragraphs (rough check)
python3 -c "
import glob
for p in sorted(glob.glob('/root/hwpx_verify/Contents/section*.xml')):
    with open(p) as f:
        content = f.read()
    if 'lineSegArray' in content.lower():
        print(f'WARNING: lineSegArray still present in {p}')
    else:
        print(f'OK: no lineSegArray in {p}')
"
```

If any assertion fails, inspect the actual XML, fix the replacement logic, and re-run.

## 5 — Confirm final file exists
```bash
ls -la /root/renewal_playbook_updated.hwpx
```

## Key Pitfalls to Avoid
- **Split tags**: Text in HWPX is often split across multiple `<hp:t>` elements within a paragraph. Always concatenate all `<hp:t>` text within a `<hp:p>` before searching, then rewrite into the first `<hp:t>` and clear the rest.
- **Layout cache**: Always remove `<hp:lineSegArray>` (and any similar layout-cache elements like `<hp:lineseg>`) from any paragraph whose text content was changed.
- **ZIP structure**: Preserve the original ZIP member order and include all original files (not just modified ones).
- **Old values**: Ensure old values are fully removed, not left alongside new values.
- **Namespace awareness**: Use proper namespace handling with lxml. The HWPX namespace for `hp:` elements must be correctly resolved.
- **renewal_update.json structure**: Inspect the actual JSON structure carefully before assuming key names like 'old'/'new'. Adapt the script to the actual schema.
- **followups.csv structure**: Inspect actual column headers. The follow-up text to insert may be in a column like 'action', 'description', 'text', etc. Also check if the full line (combining multiple columns) needs to be inserted.

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