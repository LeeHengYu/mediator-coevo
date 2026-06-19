# Task Instruction

Complete the following task to update a HWPX supplier contact sheet template with actual data.

## Background
HWPX files are ZIP-based document packages (similar to DOCX) used by the Korean word processor Hancom/Hangul. Inside the ZIP, the main document content is typically in XML files (often `Contents/section0.xml` or similar). You need to:
1. Replace template placeholders with real values from a JSON file
2. Clean up layout cache elements so the document renders correctly
3. Repackage everything as a valid HWPX (ZIP) file

## Steps

### Step 1: Examine the workspace
```
ls /root/
cat /root/supplier_contact.json
```
Understand the JSON structure and all the key-value pairs available.

### Step 2: Explore the HWPX template structure
```
mkdir -p /tmp/hwpx_work
cp /root/supplier_contact_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip -o template.zip -d template_contents
find template_contents -type f | sort
```
List all files inside the HWPX package to understand its structure.

### Step 3: Find all placeholders
Search every XML/text file inside the extracted package for `{{` patterns:
```
grep -r '{{' template_contents/
```
Note every placeholder found (e.g., `{{company_name}}`, `{{contact_person}}`, etc.) and map each to the corresponding JSON key.

### Step 4: Examine the XML structure in detail
Read the files containing placeholders carefully:
```
cat template_contents/Contents/section0.xml
```
(or whatever path contains the placeholders). Pay special attention to:
- The XML namespace declarations
- How text runs are structured (placeholders might be split across multiple XML text nodes within a single paragraph)
- Any layout cache elements (`<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:lineseg>`, `<hp:lineSeg>`, or similar elements that cache glyph/line layout information)

**CRITICAL**: Placeholders like `{{company_name}}` might be split across multiple `<hp:t>` elements within the same paragraph (e.g., one element has `{{company` and another has `_name}}`). You MUST handle this by examining the raw XML carefully.

### Step 5: Write a Python script to perform the replacement
Create `/tmp/hwpx_work/process.py` that:

a) Reads `supplier_contact.json`
b) For each XML file containing placeholders:
   - Reads the raw XML content
   - **First pass**: Concatenate all text content within each paragraph to detect split placeholders. If a placeholder is split across multiple `<hp:t>` (or equivalent) text elements within the same run/paragraph, merge them appropriately so the placeholder can be matched and replaced.
   - **Second pass**: Replace every `{{key}}` placeholder with the corresponding value from the JSON
   - **Third pass**: Remove or empty out layout cache elements. Specifically, for any paragraph (`<hp:p>`) whose text content was modified, remove the contents of `<hp:linesegarray>` or `<hp:lineSegArray>` elements (delete all child `<hp:lineseg>`/`<hp:lineSeg>` elements within them, or remove the linesegarray element entirely). This prevents stale layout cache from causing overlapping characters.
   - Verify no `{{` patterns remain in the output
c) Writes the modified XML back

Use Python's `xml.etree.ElementTree` or `lxml` if available, but be very careful with namespaces. If namespace handling is complex, consider using regex-based processing on the raw XML string — but preserve XML structure integrity.

**Important considerations for the script**:
- Preserve all Korean field labels (do not modify text that isn't a placeholder)
- Preserve the static note line unchanged
- Handle the case where a JSON value might need to go into a specific text node
- After all replacements, do a final scan: `grep -c '{{' output_file` must return 0

### Step 6: Run the script
```
cd /tmp/hwpx_work
python3 process.py
```

### Step 7: Verify no placeholders remain
```
grep -r '{{' template_contents/
```
This must return nothing. If any placeholders remain, fix the script and re-run.

### Step 8: Repackage as HWPX
The HWPX file must be a valid ZIP. Repackage from the extracted directory:
```
cd /tmp/hwpx_work/template_contents
zip -r /root/supplier_contact_ready.hwpx . -x '*.DS_Store'
```
Note: Use `zip -0 -r` if the original used stored (uncompressed) entries, or just `zip -r` for standard compression. Check the original:
```
unzip -l /tmp/hwpx_work/template.zip | head -20
```
to see if entries were stored or deflated, and match that behavior.

### Step 9: Validate the output
```
# Verify it's a valid ZIP
unzip -t /root/supplier_contact_ready.hwpx

# Verify no placeholders remain
unzip -p /root/supplier_contact_ready.hwpx | grep -c '{{' || true

# Verify the file exists and has reasonable size
ls -la /root/supplier_contact_ready.hwpx
```

### Step 10: Final content check
Extract and display the main content XML from the output file to visually confirm:
- All placeholders are replaced with actual values
- Korean labels are preserved
- The static note line is unchanged
- Layout cache elements in modified paragraphs are cleaned

```
mkdir -p /tmp/verify
cd /tmp/verify
unzip -o /root/supplier_contact_ready.hwpx
cat Contents/section0.xml
```
(adjust path as needed based on actual structure)

Do NOT consider the task complete until all verifications pass.

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