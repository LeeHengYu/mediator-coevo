# Task Instruction

Complete the project proposal document by following these steps precisely:

## Step 1: Inspect the workspace
```bash
ls /root/
find /root/ -name 'project_proposal_template.hwpx' -o -name 'project_proposal.json' 2>/dev/null
```
Identify the exact paths of both input files.

## Step 2: Read the JSON data
```bash
cat /root/project_proposal.json
```
Note every key-value pair. For the budget value, plan to strip commas but keep the leading currency symbol (₩).

## Step 3: Explore the HWPX package structure
An HWPX file is a ZIP archive. Extract it to a temp directory:
```bash
mkdir -p /tmp/hwpx_work
cd /tmp/hwpx_work
cp /root/project_proposal_template.hwpx /tmp/hwpx_work/template.hwpx
python3 -c "
import zipfile, os
with zipfile.ZipFile('template.hwpx', 'r') as z:
    z.extractall('extracted')
    for f in z.namelist():
        print(f)
"
```
List all files in the archive. Identify XML section files (typically under `Contents/` with `.xml` extension).

## Step 4: Inspect all XML content files for placeholders
For every XML file found in the archive, print its contents and search for `{{` patterns:
```bash
grep -rn '{{' /tmp/hwpx_work/extracted/
```
Also do a full read of each section XML to understand the paragraph structure, especially `<hp:t>` text nodes, `<hp:run>` elements, and `<hp:lineSegArray>` layout cache elements.

## Step 5: Write a Python script to perform all edits
Create `/tmp/hwpx_work/process.py` that does the following:

### 5a. Load JSON values
Read `project_proposal.json`. For the budget key, remove commas from the numeric part but preserve the ₩ symbol.

### 5b. For each XML content file in the extracted archive:
1. Parse the XML preserving namespaces (use `lxml.etree` or `xml.etree.ElementTree` with namespace handling). Register all namespaces before parsing to avoid ns0/ns1 prefix rewriting.
2. **Merge fragmented text nodes**: For each paragraph element (`<hp:p>`), collect all `<hp:t>` text nodes in order, concatenate their text into a single string. This is critical because `{{placeholder}}` text is often split across multiple `<hp:t>` tags.
3. **Replace placeholders**: In the merged paragraph text, replace every `{{key}}` with the corresponding JSON value (budget already normalized). Use regex `r'\{\{([^}]+)\}\}'` to find all placeholders.
4. **Append month spans for phase lines**: After placeholder replacement, check if the paragraph text contains `단계1`, `단계2`, or `단계3`. If it does, parse the date range already present in that line (format like `2025.01~2025.03` or `2025.01-2025.03` or similar). Calculate the inclusive month span: `(end_year - start_year) * 12 + (end_month - start_month) + 1`. Append ` (N개월)` to the paragraph text. Expected: 단계1 → (3개월), 단계2 → (3개월), 단계3 → (1개월).
5. **Rebuild the paragraph's run/text nodes**: After determining the final text for a modified paragraph, set the first `<hp:t>` element's text to the full merged+replaced string, and remove all subsequent `<hp:t>` elements (and their parent `<hp:run>` if they become empty). This prevents duplicate/fragmented text.
6. **Remove stale layout cache**: For every paragraph that was modified, find and remove all `<hp:lineSegArray>` child elements (and `<hp:lineSeg>` if structured differently). This prevents overlapping characters when the document is opened.
7. **Preserve everything else**: Do not modify Korean labels, static note lines, or any paragraph that doesn't contain placeholders or phase lines.

### 5c. Write modified XML back
Serialize the XML back to the file, preserving the XML declaration and encoding.

### 5d. Repackage the HWPX
Create the output HWPX by zipping the extracted directory back, preserving the original archive structure (same filenames, same directory layout). Use `zipfile.ZipFile` with `ZIP_DEFLATED`. Write to `/root/project_proposal_ready.hwpx`.

## Step 6: Run the script
```bash
cd /tmp/hwpx_work
python3 process.py
```

## Step 7: Validate the output
1. Verify the output is a valid ZIP:
```bash
python3 -c "import zipfile; z=zipfile.ZipFile('/root/project_proposal_ready.hwpx'); print(z.namelist()); z.close()"
```
2. Check no `{{` placeholders remain:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('/root/project_proposal_ready.hwpx') as z:
    for name in z.namelist():
        try:
            content = z.read(name).decode('utf-8', errors='ignore')
            if '{{' in content:
                print(f'PLACEHOLDER FOUND in {name}:', [line for line in content.split('\\n') if '{{' in line])
        except: pass
    else:
        print('No placeholders found - PASS')
"
```
3. Check month spans are present:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('/root/project_proposal_ready.hwpx') as z:
    for name in z.namelist():
        try:
            content = z.read(name).decode('utf-8', errors='ignore')
            for term in ['3개월', '1개월']:
                if term in content:
                    print(f'{term} found in {name}')
        except: pass
"
```
4. Verify budget value has no commas but has ₩:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('/root/project_proposal_ready.hwpx') as z:
    for name in z.namelist():
        try:
            content = z.read(name).decode('utf-8', errors='ignore')
            if '₩' in content:
                import re
                matches = re.findall(r'₩[\d,]+', content)
                for m in matches:
                    print(f'Budget value: {m}')
                    if ',' in m:
                        print('ERROR: comma still present in budget')
                    else:
                        print('OK: no commas')
        except: pass
"
```
5. Verify no `lineSegArray` remains in modified paragraphs (spot check).

## Critical Notes
- **Namespace handling is crucial**: When parsing HWPX XML, register namespaces before parsing to avoid rewriting prefixes. Use `lxml` if available, otherwise carefully handle with `xml.etree.ElementTree`.
- **Fragmented `<hp:t>` tags**: The most common failure mode. Always merge all text within a paragraph before doing regex replacement.
- **lineSegArray removal**: Must be done for every paragraph whose text content changed, or the document will display with overlapping characters.
- **ZIP structure**: The output must mirror the original archive's file structure exactly.
- If the verifier test file exists (e.g., `test_output.py`), run it at the end:
```bash
cd /root && python3 -m pytest test_output.py -v 2>&1 | tail -40
```

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