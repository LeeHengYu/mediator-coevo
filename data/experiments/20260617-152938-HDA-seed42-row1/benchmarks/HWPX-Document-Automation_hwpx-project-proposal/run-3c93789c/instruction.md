# Task Instruction

You need to complete a project proposal HWPX document by filling in placeholders from a JSON file and performing some additional transformations.

## Steps

### 1. Inspect the workspace
```bash
ls /root/
ls /root/project_proposal.json 2>/dev/null || true
find / -name 'project_proposal_template.hwpx' 2>/dev/null
find / -name 'project_proposal.json' 2>/dev/null
```

### 2. Read the JSON data
```bash
cat project_proposal.json
```
Note every key-value pair. You will need them for placeholder replacement.

### 3. Examine the HWPX template structure
HWPX files are ZIP archives containing XML files. Unzip and inspect:
```bash
mkdir -p /tmp/hwpx_template
cd /tmp/hwpx_template
python3 -c "
import zipfile, os
with zipfile.ZipFile('/root/project_proposal_template.hwpx', 'r') as z:
    z.extractall('/tmp/hwpx_template')
    for name in z.namelist():
        print(name)
"
```

### 4. Inspect all section XML files for placeholders
Look at every XML file, especially section files (e.g., `Contents/section0.xml`, `Contents/section1.xml`, etc.):
```bash
for f in $(find /tmp/hwpx_template -name '*.xml'); do
    echo "=== $f ==="
    cat "$f"
    echo
done
```
Identify:
- All `{{...}}` placeholders and which JSON keys they map to
- Phase lines (단계1, 단계2, 단계3) with date ranges — you need to compute month spans and append them
- Budget values that need comma removal (keep currency symbol like ₩)
- Any Korean labels and static note lines that must remain unchanged

### 5. Write a Python script to produce the output
Create and run a Python script that:

a) **Reads the JSON file** to get replacement values.

b) **Opens the template HWPX as a ZIP**, iterates through all entries.

c) **For each section XML file** (files under `Contents/` ending in `.xml`):
   - Parse with `xml.etree.ElementTree` with proper namespace handling for HWPX namespaces (look for namespace declarations in the XML files — typically `hp`, `hpb`, `hc`, etc.).
   - Walk all text-bearing elements. For each element's `.text` and `.tail` attributes:
     1. Replace all `{{placeholder}}` patterns with the corresponding JSON values.
     2. For budget values: remove commas from the number but keep the currency symbol (e.g., `₩500,000,000` → `₩500000000`).
     3. For phase lines containing date ranges: compute the month span from the dates in the line and append it in parentheses. For example, if a line says `단계1: 2025.01 ~ 2025.03` and after placeholder filling it has a date range, parse the start/end dates, compute months difference, and append ` (3개월)`. Do this for 단계1, 단계2, 단계3.
   - **Remove layout cache elements** (`<hp:lineSegArray>` or similar `lineSegArray` elements) from any paragraph (`<hp:p>`) that was modified. This is critical — stale layout caches cause overlapping text in HWP viewers.
   - Serialize the modified XML back.

d) **Reassemble the HWPX ZIP**:
   - The `mimetype` file MUST be the first entry and stored with `ZIP_STORED` (no compression).
   - All other files use `ZIP_DEFLATED`.
   - Write to `/root/project_proposal_ready.hwpx`.

e) **Handle placeholders that may be split across XML nodes**: Check if any `{{` appears in one text node and `}}` in a sibling or child. If so, concatenate adjacent text runs within the same paragraph, do the replacement, then redistribute. However, based on prior success, direct per-node replacement is likely sufficient — but verify by checking for leftover `{{` after processing.

### 6. Validate the output
```bash
# Check it's a valid ZIP
python3 -c "
import zipfile
with zipfile.ZipFile('/root/project_proposal_ready.hwpx', 'r') as z:
    print('Valid ZIP, entries:', len(z.namelist()))
    for name in z.namelist():
        print(name)
"

# Check no placeholders remain
python3 -c "
import zipfile
with zipfile.ZipFile('/root/project_proposal_ready.hwpx', 'r') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            if '{{' in content or '}}' in content:
                print(f'FAIL: Placeholder remains in {name}')
                import re
                for m in re.finditer(r'\{\{.*?\}\}', content):
                    print(f'  Found: {m.group()}')
            else:
                print(f'OK: {name}')
"

# Check month spans are present
python3 -c "
import zipfile
with zipfile.ZipFile('/root/project_proposal_ready.hwpx', 'r') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            for term in ['3개월', '1개월']:
                if term in content:
                    print(f'Found {term} in {name}')
"

# Check budget has no commas
python3 -c "
import zipfile, re
with zipfile.ZipFile('/root/project_proposal_ready.hwpx', 'r') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            # Look for currency symbol followed by digits with commas
            if re.search(r'₩[\d,]*,', content):
                print(f'FAIL: Budget still has commas in {name}')
            elif '₩' in content:
                print(f'OK: Budget in {name} has no commas')
"

# Verify mimetype is first entry and stored
python3 -c "
import zipfile
with zipfile.ZipFile('/root/project_proposal_ready.hwpx', 'r') as z:
    first = z.namelist()[0]
    info = z.getinfo(first)
    print(f'First entry: {first}, compression: {info.compress_type}')
    assert first == 'mimetype', f'First entry should be mimetype, got {first}'
    assert info.compress_type == 0, 'mimetype should be ZIP_STORED'
    print('Mimetype check passed')
"
```

### 7. Check for lineSegArray removal in modified paragraphs
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('/root/project_proposal_ready.hwpx', 'r') as z:
    for name in z.namelist():
        if name.endswith('.xml') and 'section' in name.lower():
            content = z.read(name).decode('utf-8')
            if 'lineSegArray' in content:
                print(f'WARNING: lineSegArray still present in {name} — check if only in unmodified paragraphs')
            else:
                print(f'OK: No lineSegArray in {name}')
"
```

### Key Reminders
- **Month span calculation**: Parse dates like `2025.01 ~ 2025.03`. Compute: `(end_year - start_year) * 12 + (end_month - start_month)` months. Append as ` (N개월)` after the phase line text.
- **Budget normalization**: Remove commas from numeric part, keep ₩ symbol.
- **Namespace handling**: Register all namespaces found in the XML before parsing to avoid namespace prefix mangling in output.
- **Do NOT modify Korean labels or the static note line.**
- **Ensure the output file is at exactly `/root/project_proposal_ready.hwpx`.**

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