# Task Instruction

Complete the following task to fill in a HWPX supplier contact sheet template.

## Goal
Replace all `{{...}}` placeholders in `supplier_contact_template.hwpx` with values from `supplier_contact.json`, and save the result to `/root/supplier_contact_ready.hwpx`.

## Steps

### 1. Inspect the input files
- Read `supplier_contact.json` to understand the keys and values available.
- Since `.hwpx` is a ZIP archive, list its contents: `python3 -c "import zipfile; z=zipfile.ZipFile('supplier_contact_template.hwpx'); print('\n'.join(z.namelist()))"`
- Extract and read the XML content files inside the HWPX archive. The main document content is typically in files like `Contents/section0.xml` or similar paths. Read ALL XML files in the archive to find where placeholders exist.
- Search for ALL occurrences of `{{` across every file in the archive to identify every placeholder location.

### 2. Understand placeholder structure
- CRITICAL: In XML-based word processor formats, a single placeholder like `{{company_name}}` may be split across multiple XML text runs/elements. For example, `{{` might be in one `<hp:t>` element and `company_name}}` in another, or even more fragmented.
- Read the raw XML carefully to understand how placeholders are distributed across elements.
- Build a strategy: concatenate text from consecutive runs within a paragraph, find placeholders in the concatenated text, then reconstruct the runs with the replacement values.

### 3. Understand layout cache elements
- Look for elements related to layout caching in the XML. These might be elements like `<hp:linesegarray>`, `<hp:lineseg>`, `<hp:lineSegArray>`, `<hp:LineSeg>`, or similar elements that cache glyph positions and line layout.
- Any paragraph whose text content is modified MUST have these layout-cache elements removed so the word processor recalculates layout on open.

### 4. Write a Python script to perform the replacement
Write a Python script (`/root/fill_template.py`) that:

a) Reads `supplier_contact.json` to get the replacement values.

b) Opens `supplier_contact_template.hwpx` as a ZIP archive.

c) For each file in the archive:
   - If it's an XML content file (especially section XML files), parse it and handle placeholder replacement.
   - For other files (images, settings, etc.), copy them as-is.

d) For placeholder replacement in XML:
   - Parse the XML with `lxml.etree` or `xml.etree.ElementTree`.
   - For each paragraph element, collect all text content from its child text run elements in order.
   - Concatenate the text to find `{{key}}` patterns.
   - Map each placeholder to its JSON value using the key.
   - Redistribute the replaced text back into the run elements (simplest approach: put all replaced text into the first run's text element and clear the others, or reconstruct appropriately).
   - If any replacement was made in a paragraph, remove all layout-cache child elements from that paragraph (elements like `linesegarray`, `LineSeg`, `lineSegArray`, or similar - identify the exact element names from the XML inspection).

e) Write the output as a new ZIP file at `/root/supplier_contact_ready.hwpx`, preserving the same archive structure and compression.

### 5. Handle XML namespaces carefully
- HWPX XML files use namespaces. When searching for elements, use the correct namespace URIs.
- When writing back XML, preserve all namespace declarations.
- Use `lxml` if available for better namespace handling; fall back to `xml.etree.ElementTree` if needed.
- When serializing XML back, preserve the XML declaration and encoding.

### 6. Run the script and validate
- Run: `python3 /root/fill_template.py`
- Validate the output exists: `ls -la /root/supplier_contact_ready.hwpx`
- Validate it's a valid ZIP: `python3 -c "import zipfile; print(zipfile.is_zipfile('/root/supplier_contact_ready.hwpx'))"`
- Verify NO `{{` placeholders remain anywhere: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/supplier_contact_ready.hwpx'); [print(f'PLACEHOLDER FOUND in {n}: {z.read(n).decode(errors=\"replace\")}') for n in z.namelist() if b'{{' in z.read(n)]"` — this should produce NO output.
- Extract and print the content XML from the output to visually confirm the Korean labels are preserved and values are filled in correctly.
- Confirm layout-cache elements were removed from modified paragraphs.

### 7. Important constraints
- Do NOT remove or alter Korean field labels (e.g., 회사명, 담당자, etc.).
- Do NOT alter the static note line in the document.
- Every single `{{...}}` placeholder must be replaced — verify exhaustively.
- The file must remain a valid `.hwpx` ZIP package with the same internal structure.

### 8. Debugging
- If placeholders span multiple XML runs, the concatenation approach is essential. Do not try simple string replacement on individual text elements as it will miss split placeholders.
- If `lxml` is not available, use `xml.etree.ElementTree` but be careful with namespace handling.
- If the output ZIP is corrupted, check compression method and ensure binary files are copied correctly.

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