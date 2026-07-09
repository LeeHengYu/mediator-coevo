# Task Instruction

## Task: Update renewal_playbook.hwpx with new data

You need to revise `renewal_playbook.hwpx` using `renewal_update.json` and `followups.csv`, saving the result to `/root/renewal_playbook_updated.hwpx`.

### Step-by-step plan:

#### 1. Inspect available files
```bash
ls -la /root/
file /root/renewal_playbook.hwpx
cat /root/renewal_update.json
cat /root/followups.csv
```

#### 2. Explore the .hwpx package structure
A .hwpx file is a ZIP archive containing XML files. Extract and examine:
```bash
mkdir -p /tmp/hwpx_orig
cd /tmp/hwpx_orig
python3 -c "import zipfile; z=zipfile.ZipFile('/root/renewal_playbook.hwpx'); z.printdir(); z.extractall('.')"
find . -type f
```
Read each XML file to understand the document structure. Pay special attention to:
- The main content XML (likely `Contents/section0.xml` or similar)
- Any manifest or metadata files
- The encoding and namespace declarations

#### 3. Understand the data to update
From `renewal_update.json`, identify the fields and their old→new values. The JSON likely contains new values for:
- Customer name
- Current owner
- Renewal window
- Pricing band
- Escalation contact
- Pricing note

You need to find the OLD values by reading the current document XML, then replace them with the NEW values from the JSON.

#### 4. Understand follow-up replacements
From `followups.csv`, read the items and sort by the `sequence` column. These replace the existing three follow-up lines in the document.

#### 5. Write a Python script to perform the update
Create a comprehensive Python script that:

a. **Extracts** the .hwpx ZIP to a temp directory
b. **Reads** the JSON and CSV data
c. **Parses** all XML content files using `xml.etree.ElementTree` (preserving namespaces)
d. **Identifies old values** in the XML text content by reading the original document first
e. **Replaces** customer name, owner, renewal window, pricing band, escalation contact, and pricing note everywhere they appear in editable sections
f. **Replaces** the three follow-up lines with CSV items ordered by sequence
g. **Removes stale layout-cache elements**: For any `<hp:linesegarray>` or similar layout-cache/line-segment elements within paragraphs whose text was modified, remove those elements entirely. Look for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, or any element that serves as a layout cache. These are typically child elements of paragraph (`<hp:p>`) elements.
h. **Preserves** the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` exactly as-is
i. **Repackages** everything back into a valid ZIP with `.hwpx` extension

#### 6. Important XML handling details
- Register all namespaces before parsing to avoid `ns0:` prefix pollution. Use `ET.register_namespace()` for each namespace found in the XML.
- When searching for text, check all text-bearing elements (look for elements containing actual text runs, often `<hp:t>` or similar)
- The .hwpx format uses Korean OASIS-like XML namespaces - inspect them carefully
- Preserve the original ZIP compression method and structure

#### 7. Layout-cache cleanup
After modifying any paragraph's text content, find and remove layout-cache child elements from that paragraph. Common layout-cache element names in HWPX:
- `linesegarray` / `lineSegArray`
- `lineseg` / `lineSeg`  
- Any element whose local name contains 'cache', 'seg', or 'layout' that is a child of a modified paragraph

Inspect the actual XML to determine the exact element names before writing removal code.

#### 8. Repackage the .hwpx
When creating the output ZIP:
- Use `zipfile.ZipFile` with `ZIP_DEFLATED` compression
- Preserve the exact directory structure from the original
- Write to `/root/renewal_playbook_updated.hwpx`

#### 9. Validate the output
```bash
python3 -c "import zipfile; z=zipfile.ZipFile('/root/renewal_playbook_updated.hwpx'); z.printdir(); z.testzip()"
```
Also verify:
- The appendix sentence is preserved: search for `이 부록 문단은 그대로 유지해야 합니다.` in the output XML
- The old values no longer appear in the document
- The new values from JSON appear in the document
- The follow-up lines from CSV appear in sequence order
- No stale layout-cache elements remain in modified paragraphs
- The file is a valid ZIP

### Critical reminders:
- Do NOT add new content alongside old content; REPLACE old values, removing them entirely
- The follow-up lines must be in `sequence` order from the CSV
- Every paragraph you modify must have its layout-cache elements removed
- The Korean appendix sentence must remain byte-for-byte identical
- Read and understand the actual XML structure before writing any modification code

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