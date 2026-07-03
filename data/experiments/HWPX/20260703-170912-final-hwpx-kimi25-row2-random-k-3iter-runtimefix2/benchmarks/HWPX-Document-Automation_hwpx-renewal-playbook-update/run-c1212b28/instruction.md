# Task Instruction

Complete the following task to update a HWPX document:

## Goal
Revise `renewal_playbook.hwpx` using `renewal_update.json` and `followups.csv`, saving the result to `/root/renewal_playbook_updated.hwpx`.

## Step-by-step Plan

### Step 1: Explore the workspace and understand inputs
1. Run `find /root -maxdepth 2 -type f` to locate all relevant files.
2. Read `renewal_update.json` completely — note every field name and its old/new value.
3. Read `followups.csv` completely — note the columns, especially a `sequence` column that determines ordering.
4. List the contents of `renewal_playbook.hwpx` by running `python3 -c "import zipfile; z=zipfile.ZipFile('renewal_playbook.hwpx'); print('\n'.join(z.namelist()))"` (adjust path as needed).
5. Extract the hwpx to a temp directory: `mkdir /tmp/hwpx && cd /tmp/hwpx && python3 -c "import zipfile; zipfile.ZipFile('/root/renewal_playbook.hwpx').extractall('/tmp/hwpx')"` (adjust source path).
6. Examine ALL XML files inside, especially the main content XML (likely `Contents/section0.xml` or similar). Read each XML file fully to understand the document structure.

### Step 2: Understand the HWPX XML structure
1. Carefully read the main section XML file(s). Identify:
   - Where the customer name, current owner, renewal window, pricing band, escalation contact, and pricing note appear in text runs.
   - Where the three follow-up lines are (look for numbered/sequential follow-up items).
   - Where the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` appears.
   - Any layout-cache elements (look for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<lineseg>`, `<hp:lineseg>`, or similar cache/layout elements within paragraph nodes).
2. Print the full content of each XML file — do NOT skip or truncate. You need to see every character.

### Step 3: Write a Python script to perform the update
Write a single Python script `/tmp/update_hwpx.py` that:

1. **Extracts** the hwpx ZIP to a temp directory.
2. **Reads** `renewal_update.json` to get the mapping of old values → new values for each field. The JSON likely has keys like `customer_name`, `current_owner`, `renewal_window`, `pricing_band`, `escalation_contact`, `pricing_note` with old and new values (or just new values — inspect the actual JSON structure first).
3. **Reads** `followups.csv`, sorts by the `sequence` column, and gets the ordered list of new follow-up lines.
4. **Parses** the section XML file(s) using `xml.etree.ElementTree` with proper namespace handling.
5. **For each field to update**: Finds all text nodes containing the old value and replaces with the new value. Be careful with namespaces — register them properly so they are preserved in output.
6. **For follow-up lines**: Identifies the existing three follow-up paragraphs (by their text content or structure), replaces their text content with the CSV items in sequence order. If there are exactly 3 old follow-ups and potentially a different number of new ones, handle accordingly.
7. **Removes layout-cache elements** from any paragraph whose text was modified. Look for elements like `linesegarray`, `lineSegArray`, or similar within `<hp:p>` or `<p>` elements. After modifying text in a paragraph, find and remove these cache child elements.
8. **Verifies** the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is still present and unchanged.
9. **Writes** the modified XML back, preserving XML declarations, namespaces, and encoding.
10. **Re-packages** everything into a new ZIP file at `/root/renewal_playbook_updated.hwpx`, preserving the original ZIP structure (same file paths, mimetype handling, etc.).

### Step 4: Important XML/namespace considerations
- Before writing the script, identify ALL namespace URIs used in the XML files. Register them with `ET.register_namespace()` so they are preserved.
- Use `ET.parse()` not `ET.fromstring()` to preserve XML declarations.
- When writing back, use the same encoding as the original.
- Preserve the `mimetype` file if present (should be first in ZIP, uncompressed if originally so).

### Step 5: Execute and validate
1. Run the script: `python3 /tmp/update_hwpx.py`
2. Verify the output exists: `ls -la /root/renewal_playbook_updated.hwpx`
3. Verify it's a valid ZIP: `python3 -c "import zipfile; print(zipfile.is_zipfile('/root/renewal_playbook_updated.hwpx'))"`
4. Extract and inspect the modified XML to confirm:
   - All old field values are gone (search for them)
   - All new field values are present
   - Follow-up lines match CSV in sequence order
   - The appendix sentence is preserved exactly
   - No layout-cache elements remain on modified paragraphs
   - Layout-cache elements on UNmodified paragraphs are still intact
5. Compare file lists of original and updated hwpx to ensure no files were lost or added unexpectedly.

### Critical Details
- The old values to search-and-replace must come from inspecting the ACTUAL document content, not from assumptions. Read the XML first, then determine what text to replace.
- If `renewal_update.json` contains both old and new values, use them directly. If it only contains new values, you must extract old values from the document itself by understanding which field maps to which text location.
- Handle Korean text (UTF-8) properly throughout.
- Do NOT modify the appendix paragraph at all — not its text, not its attributes, not its layout cache.
- When removing layout cache elements, only remove them from paragraphs where you actually changed text content.

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