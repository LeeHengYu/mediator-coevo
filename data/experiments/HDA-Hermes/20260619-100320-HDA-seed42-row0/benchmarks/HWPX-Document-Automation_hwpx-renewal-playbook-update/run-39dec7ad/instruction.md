# Task Instruction

Complete the following task to update a HWPX document. Follow each step carefully.

## Goal
Revise `renewal_playbook.hwpx` using data from `renewal_update.json` and `followups.csv`, saving the result to `/root/renewal_playbook_updated.hwpx`.

## Steps

### Step 1: Examine the input files
1. Find the working directory containing `renewal_playbook.hwpx`, `renewal_update.json`, and `followups.csv`. Check `/root/` and common locations.
2. Read `renewal_update.json` — note every field name and its new value.
3. Read `followups.csv` — note all rows and the `sequence` column that determines ordering.
4. Print findings before proceeding.

### Step 2: Explore the HWPX package structure
1. HWPX is a ZIP archive. List its contents: `python3 -c "import zipfile; z=zipfile.ZipFile('renewal_playbook.hwpx'); z.printdir()"`
2. Identify all XML content files (likely under `Contents/` — look for files like `section0.xml`, `content.hpf`, etc.).
3. Read and print EVERY XML file in the archive to understand the full document structure. Pay special attention to:
   - Where the editable text content lives (section XML files)
   - The current customer name, owner, renewal window, pricing band, escalation contact, pricing note
   - The three existing follow-up lines
   - The appendix sentence `이 부록 문단은 그대로 유지해야 합니다.`
   - Any layout-cache elements (look for tags containing `lineseg`, `lineSegArray`, `charPr`, `cache`, or similar positioning/layout data within paragraph elements)

### Step 3: Identify old values to replace
From the JSON update file, determine what the OLD values are by reading them from the current document XML. Map each field (customer name, current owner, renewal window, pricing band, escalation contact, pricing note) to its old text string found in the document. These old strings must be replaced with the new values from the JSON.

### Step 4: Write a Python script to perform all edits
Write a single Python script `/root/update_hwpx.py` that:

1. **Copies the HWPX ZIP** to the output path, then modifies it in-place (or builds a new ZIP preserving all non-modified entries byte-for-byte).
2. **For each XML content file** in the package that contains document text:
   a. Parse the XML (use `lxml.etree` if available, otherwise `xml.etree.ElementTree` — be careful with namespaces).
   b. **Field replacements**: For every text node in the XML, replace occurrences of old field values with new values from the JSON. Do this for ALL fields: customer name, current owner, renewal window, pricing band, escalation contact, and pricing note. Replace everywhere they appear.
   c. **Follow-up replacement**: Identify the three existing follow-up line paragraphs. Remove them and insert new paragraphs (or modify their text) for each row in `followups.csv`, ordered by the `sequence` column. Match the XML structure of the original follow-up paragraphs for the new ones.
   d. **Preserve the appendix sentence** `이 부록 문단은 그대로 유지해야 합니다.` — verify it remains unchanged.
   e. **Remove layout-cache elements from modified paragraphs**: For any paragraph element whose text content was changed, find and remove child elements that represent layout caching (e.g., `lineseg`, `lineSegArray`, or similar elements that cache character positions/glyph layout). Do NOT remove these from unmodified paragraphs. This is critical for the document to open without overlapping characters.
   f. Serialize the modified XML back, preserving the XML declaration and encoding.
3. **Write the final ZIP** to `/root/renewal_playbook_updated.hwpx`, ensuring all other files from the original package are preserved exactly.

### Step 5: Execute and validate
1. Run the script: `python3 /root/update_hwpx.py`
2. Verify the output is a valid ZIP: `python3 -c "import zipfile; print(zipfile.is_zipfile('/root/renewal_playbook_updated.hwpx'))"`
3. List the contents of the output ZIP and compare with the original — same files should be present.
4. Extract and print the modified XML content files from the output ZIP. Verify:
   - All old field values are gone (search for them explicitly)
   - All new field values from the JSON are present
   - Follow-up lines match the CSV data in sequence order
   - The appendix sentence is preserved exactly
   - Layout-cache elements are removed from modified paragraphs but preserved in unmodified ones
   - No duplicate lines exist (old values alongside new values)
5. If any check fails, fix the script and re-run.

### Important Technical Notes
- HWPX XML uses namespaces extensively. When searching/modifying, handle namespaces properly. Print the root element tag to discover the namespace URI.
- The follow-up replacement must maintain valid XML structure — don't just do string replacement for these; manipulate the XML tree.
- For field text replacements, text may be split across multiple `<hp:t>` or similar text run elements within a paragraph. Check if values span multiple runs and handle accordingly.
- When removing layout-cache elements, look for elements like `{namespace}lineseg` or `{namespace}lineSegArray` that are direct children of paragraph elements. Print examples of these elements before removing them so you understand their structure.
- Ensure the output ZIP uses the same compression method as the original for each entry.

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