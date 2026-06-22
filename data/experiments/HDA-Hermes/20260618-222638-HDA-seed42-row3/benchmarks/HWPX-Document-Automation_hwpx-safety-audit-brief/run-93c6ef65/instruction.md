# Task Instruction

## Task: Prepare warehouse safety audit brief HWPX document

### Goal
Fill the template `safety_audit_template.hwpx` with data from `audit_overview.json` and `corrective_actions.json`, then save the result to `/root/safety_audit_brief_final.hwpx`.

### Step-by-step plan

#### 1. Explore the workspace
```bash
cd /root
find . -maxdepth 2 -type f | head -60
ls -la *.json *.hwpx 2>/dev/null || true
```
Identify where the template and JSON files live (likely under a task directory).

#### 2. Read the JSON data files
```bash
cat audit_overview.json
cat corrective_actions.json
```
Note every field name and value. Pay special attention to:
- The **risk tier** value (e.g., `High`, `Medium`, or `Low`)
- The **inspection date** in `YYYY-MM-DD` format
- All overview fields and audit-table values
- The three corrective-action entries and their order

#### 3. Unpack and inspect the HWPX template
HWPX is a ZIP-based package. Unpack it to understand its structure:
```bash
mkdir -p /tmp/hwpx_work
cp safety_audit_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_contents
find template_contents -type f
```
Then read every XML file that contains document content (typically under `Contents/` — look for `section0.xml`, `section1.xml`, etc.):
```bash
for f in $(find template_contents -name '*.xml'); do echo "=== $f ==="; cat "$f"; echo; done
```
Also check `content.hpf` or `content.xml` or any manifest file.

#### 4. Identify all placeholders
Search for `{{` patterns across all XML files:
```bash
grep -rn '{{' template_contents/
```
Make a complete list of every `{{...}}` placeholder and which file/line it appears in.

#### 5. Write a Python script to perform all substitutions

Create `/tmp/hwpx_work/build.py` that does the following:

##### 5.1 Load JSON data
Read both JSON files.

##### 5.2 Date reformatting
Convert the inspection date from `YYYY-MM-DD` to `YYYY.MM.DD`. Apply this transformation to every occurrence of the date in the XML content (both the `YYYY-MM-DD` form and any placeholder that resolves to it).

##### 5.3 Risk tier + severity note
Map the risk tier to a severity note:
- `High` → `즉시조치`
- `Medium` → `계획보완`
- `Low` → `모니터링`

**CRITICAL FORMAT**: The severity note MUST be appended in parentheses with a space before the opening parenthesis. Example: `High (즉시조치)`. The test suite asserts exactly this format. Do NOT use `High 즉시조치` or `High - 즉시조치` or any other variant.

When substituting the risk tier placeholder, replace it with `{tier} ({note})`. Also search for any other literal occurrences of the risk tier string and replace them with the annotated version.

##### 5.4 Fill all placeholders
For each `{{...}}` placeholder found in step 4, substitute the corresponding value from the JSON data. This includes:
- Overview/summary fields
- Audit table value cells
- Corrective action lines (maintain the order from `corrective_actions.json`)

##### 5.5 Remove stale layout-cache elements
For any paragraph (`<hp:p>`) whose text content was modified, remove all `<hp:linesegarray>` elements (and their children) within that paragraph. This prevents overlapping character rendering. Use an XML parser (lxml or xml.etree.ElementTree) with proper namespace handling, or use careful regex if the namespace structure is complex.

**Approach for linesegarray removal**: Parse the XML, find all `linesegarray` elements, check if they're inside paragraphs that contain modified text, and remove them. To be safe, you may remove ALL `linesegarray` elements since the document will regenerate them on open.

##### 5.6 Verify no remaining placeholders
After all substitutions, assert that no `{{` or `}}` patterns remain in any XML file. If any remain, log them and fix them.

##### 5.7 Repack the HWPX
Repack the modified files back into a ZIP with `.hwpx` extension, preserving the original directory structure and using the same compression method. Save to `/root/safety_audit_brief_final.hwpx`.

```python
import zipfile, os

def repack(source_dir, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            for fname in files:
                full = os.path.join(root, fname)
                arcname = os.path.relpath(full, source_dir)
                zf.write(full, arcname)
```

#### 6. Run and verify
```bash
python3 /tmp/hwpx_work/build.py
```

Then verify the output:
```bash
# Check it's a valid zip
unzip -t /root/safety_audit_brief_final.hwpx

# Extract and check content
mkdir -p /tmp/verify
cd /tmp/verify
unzip /root/safety_audit_brief_final.hwpx -d verify_contents

# Check no placeholders remain
grep -rn '{{' verify_contents/ && echo 'FAIL: placeholders remain' || echo 'OK: no placeholders'

# Check risk tier format
grep -rn '즉시조치\|계획보완\|모니터링' verify_contents/
# Verify the format includes parentheses, e.g., 'High (즉시조치)'

# Check date format is YYYY.MM.DD not YYYY-MM-DD
grep -rn '[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}' verify_contents/ && echo 'FAIL: old date format found' || echo 'OK: dates reformatted'

# Check linesegarray removal in modified paragraphs
grep -c 'linesegarray' verify_contents/Contents/*.xml || echo 'linesegarray elements removed'
```

#### 7. Run the test suite if available
```bash
cd /root
find . -name 'test_output*' -o -name 'test_*.py' | head -5
# If found:
python3 -m pytest test_output.py -v 2>&1 | tail -40
```
If tests fail, read the assertion errors carefully, fix the build script, and re-run.

### Key pitfalls to avoid (from prior feedback)
1. **Severity note format**: MUST be `TierName (한글note)` with parentheses. The test asserts `'High (즉시조치)' in content`.
2. **The substitution must land inside `<hp:t>` tags** in the XML — the text content tags. Make sure you're modifying the right elements.
3. **linesegarray removal**: The test checks that edited paragraphs don't retain stale layout cache. Remove `<hp:linesegarray>` elements from modified paragraphs.
4. **All occurrences**: The risk tier and date may appear multiple times. Replace ALL of them, not just the first.
5. **Corrective actions order**: Must match the order in `corrective_actions.json`.
6. **Valid HWPX package**: The output must be a proper ZIP file with the correct internal structure.

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