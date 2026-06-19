# Task Instruction

Complete the inventory status report by filling in the HWPX template with JSON data. Follow these steps precisely:

## Step 1: Inspect the workspace
```bash
ls /root/
cat /root/inventory_data.json
```
Examine the JSON data to understand all key-value pairs that will replace placeholders.

## Step 2: Examine the HWPX template structure
```bash
cd /root
mkdir -p /tmp/hwpx_work
cp inventory_report_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_extracted
find template_extracted -type f -name '*.xml' | sort
```

## Step 3: Inspect section XML files for placeholders
Look at all section XML files (typically in Contents/section*.xml) to find all `{{...}}` placeholders:
```bash
grep -r '{{' template_extracted/
```
Also cat the full section XML files to understand the structure, namespaces, and how placeholders may be split across multiple `<hp:t>` elements.

## Step 4: Write and run the Python replacement script
Create a Python script that:

1. **Loads the JSON data** from `/root/inventory_data.json`.
2. **Copies the original HWPX** (which is a ZIP) and modifies it in-place.
3. **For each XML file in the package** (especially section*.xml files):
   a. Parse the XML with a namespace-aware parser (use `lxml.etree` or `xml.etree.ElementTree`).
   b. For each paragraph element (`<hp:p>`), collect ALL text content from child `<hp:t>` elements to reconstruct the full paragraph text.
   c. Check if the reconstructed text contains any `{{key}}` patterns.
   d. If it does, perform all placeholder replacements on the reconstructed text.
   e. **Redistribute the replaced text back into the `<hp:t>` elements**: Put all the replaced text into the FIRST `<hp:t>` element and clear (or remove) subsequent `<hp:t>` elements that were part of the split placeholder. Be careful to preserve `<hp:t>` elements that belong to separate `<hp:run>` blocks if they don't participate in the placeholder.
   f. **Remove `<hp:lineSegArray>` elements** (layout cache) from any paragraph that was modified, so the document renders cleanly.
4. **Write the modified XML back** into the ZIP package.
5. **Save the result** to `/root/inventory_report_ready.hwpx`.

Key implementation details:
- Handle the case where a single placeholder like `{{report_date}}` is split across multiple `<hp:t>` nodes (e.g., `{{`, `report_date`, `}}`). The robust approach is to concatenate all `<hp:t>` text within each `<hp:run>` or paragraph, do the replacement, then redistribute.
- Preserve Korean text, empty paragraphs, and all non-placeholder content exactly.
- Use `zipfile` module to read/write the HWPX package. Create a new ZIP with the same entries, replacing only the modified XML files.
- Ensure all JSON values are converted to strings before replacement.
- After replacement, verify no `{{` or `}}` patterns remain in any XML file.

## Step 5: Validate the output
```bash
# Check it's a valid ZIP/HWPX
unzip -t /root/inventory_report_ready.hwpx

# Extract and check no placeholders remain
mkdir -p /tmp/hwpx_verify
cp /root/inventory_report_ready.hwpx /tmp/hwpx_verify/output.zip
cd /tmp/hwpx_verify
unzip output.zip -d output_extracted
grep -r '{{' output_extracted/ || echo 'No placeholders found - GOOD'

# Verify Korean labels are preserved
grep -r '보고서' output_extracted/ || true
```

## Step 6: Run the test suite
```bash
cd /root && python -m pytest test_output.py -v
```
If tests fail, read the error messages carefully, fix the issue, and re-run.

## Important Notes
- The HWPX format uses XML namespaces like `xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph"`. Make sure your XML parser handles these.
- Do NOT use regex on raw XML for replacements - parse the XML properly.
- Empty paragraphs (paragraphs with no text or only whitespace) must be preserved.
- The static note line must remain unchanged.
- Remove `hp:lineSegArray` elements ONLY from paragraphs you actually modify.

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