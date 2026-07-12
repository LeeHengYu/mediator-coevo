# Task Instruction

Complete the project proposal document by following these steps precisely:

## Step 1: Inspect the workspace
```bash
ls /root/
find /root/ -name 'project_proposal_template.hwpx' -o -name 'project_proposal.json' 2>/dev/null
```

## Step 2: Examine the JSON data
```bash
cat /root/project_proposal.json
```

## Step 3: Examine the HWPX structure
A `.hwpx` file is a ZIP archive containing XML files. Unzip it to inspect:
```bash
mkdir -p /tmp/hwpx_work
cd /tmp/hwpx_work
cp /root/project_proposal_template.hwpx /tmp/hwpx_work/template.hwpx
python3 -c "
import zipfile
with zipfile.ZipFile('template.hwpx', 'r') as z:
    z.printdir()
    z.extractall('extracted')
"
find extracted -type f
```

## Step 4: Inspect all XML content files
Read every XML file in the extracted archive, especially files under `Contents/` (like `section0.xml`, `content.hpf`, etc.). Look for:
- All `{{...}}` placeholders
- Phase lines (단계1, 단계2, 단계3) with date ranges
- Budget values
- The structure of `<hp:t>` tags and `<hp:lineSegArray>` elements

```bash
for f in $(find extracted -name '*.xml' -o -name '*.hpf'); do echo "=== $f ==="; cat "$f"; echo; done
```

## Step 5: Write and run a Python script to perform the transformation

Create a Python script `/tmp/hwpx_work/transform.py` that does the following:

1. **Load the JSON** from `/root/project_proposal.json`.
2. **Open the template HWPX** as a ZIP file.
3. **For each file in the ZIP**, if it's an XML file that may contain content:
   a. Read its text content.
   b. **Handle fragmented placeholders**: `{{placeholder_name}}` may be split across multiple `<hp:t>` tags (e.g., `<hp:t>{{</hp:t>` ... `<hp:t>name</hp:t>` ... `<hp:t>}}</hp:t>`). You must reconstruct the full text of each paragraph's `<hp:t>` elements, identify placeholders, replace them, and redistribute the text back. Use an XML parser (lxml or xml.etree.ElementTree).
   c. **Replace placeholders** with corresponding JSON values. For the budget field, remove commas from the numeric part while keeping the leading currency symbol (e.g., `₩1,000,000` → `₩1000000` or if the JSON value is `1,000,000`, output it without commas as `1000000`, preserving any `₩` or `$` prefix).
   d. **Append month spans to phase lines**: For lines containing `단계1`, `단계2`, `단계3` with date ranges, calculate the month span from the dates in that line and append ` (N개월)` after the phase content. The expected results are: 단계1 → `(3개월)`, 단계2 → `(3개월)`, 단계3 → `(1개월)`.
   e. **Remove `<hp:lineSegArray>` elements** from any paragraph (`<hp:p>`) whose text content was modified. This prevents stale layout cache from causing overlapping characters.
   f. **Verify no `{{` or `}}` remains** in the output XML.
4. **Write the new HWPX** to `/root/project_proposal_ready.hwpx` as a valid ZIP with the same structure and compression.

Key implementation details for the script:
- Parse XML with `lxml.etree` (preferred) or `xml.etree.ElementTree`.
- To handle fragmented placeholders: for each `<hp:p>` element, collect all `<hp:t>` descendants, concatenate their text, perform replacements on the concatenated string, then put the entire replaced text into the first `<hp:t>` and clear the rest (or remove extra `<hp:t>` elements and their parent `<hp:run>` if empty).
- Use namespace-aware XPath or tag matching (the namespace is typically `http://www.hancom.co.kr/hwpml/2011/paragraph` or similar — check the actual namespace in the XML files).
- For month calculation: parse dates like `2025.01 ~ 2025.03` and compute month difference.
- After all replacements, serialize the XML back preserving the XML declaration and encoding.

## Step 6: Run the script
```bash
cd /tmp/hwpx_work
python3 transform.py
```

## Step 7: Validate the output
```bash
# Check it's a valid ZIP
python3 -c "
import zipfile
with zipfile.ZipFile('/root/project_proposal_ready.hwpx', 'r') as z:
    z.printdir()
    # Extract and check for remaining placeholders
    for name in z.namelist():
        data = z.read(name)
        try:
            text = data.decode('utf-8')
            if '{{' in text or '}}' in text:
                print(f'WARNING: Remaining placeholder in {name}')
                # Print context around the placeholder
                idx = text.find('{{')
                if idx == -1: idx = text.find('}}')
                print(text[max(0,idx-50):idx+80])
        except:
            pass
print('Validation complete')
"
```

## Step 8: Verify output content
Extract the output HWPX and print the content XML files to visually confirm:
- All placeholders are replaced with correct values
- Phase lines have the correct `(N개월)` appended
- Budget has no commas
- Korean labels are preserved
- No `<hp:lineSegArray>` in modified paragraphs
- The static note line is unchanged

```bash
mkdir -p /tmp/hwpx_work/output_check
cd /tmp/hwpx_work/output_check
python3 -c "
import zipfile
with zipfile.ZipFile('/root/project_proposal_ready.hwpx', 'r') as z:
    z.extractall('.')
"
for f in $(find . -name '*.xml' -o -name '*.hpf'); do echo "=== $f ==="; cat "$f"; echo; done
```

If any issues are found (remaining placeholders, incorrect month spans, commas in budget, missing files), fix the script and re-run until the output is correct.

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