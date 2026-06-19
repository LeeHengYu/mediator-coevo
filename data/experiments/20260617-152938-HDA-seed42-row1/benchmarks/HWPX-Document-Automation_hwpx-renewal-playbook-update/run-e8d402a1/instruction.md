# Task Instruction

Execute the following steps to produce `/root/renewal_playbook_updated.hwpx`:

1. **Inspect the workspace.** List files in the task directory to locate `renewal_playbook.hwpx`, `renewal_update.json`, and `followups.csv`. Read `renewal_update.json` (JSON) and `followups.csv` (CSV) to understand the replacement values and follow-up items.

2. **Understand the HWPX structure.** A `.hwpx` file is a ZIP archive. Extract it to a temporary directory (e.g., `/tmp/hwpx_work`). List all entries. The main editable content is typically in `Contents/section0.xml` (or similar). Identify all XML files under `Contents/`.

3. **Parse the update data.**
   - From `renewal_update.json`, extract the new values for: customer name, current owner, renewal window, pricing band, escalation contact, and pricing note.
   - From `followups.csv`, read all rows and sort them by the `sequence` column (ascending). These will replace the existing three follow-up lines.

4. **Edit the section XML files.** For each section XML (likely just `section0.xml`):
   - Register all namespaces found in the file before parsing (use `xml.etree.ElementTree` with `ET.register_namespace` for each namespace prefix found, especially `hp`, `hc`, `hpf`, etc.) to avoid namespace prefix corruption on write-back.
   - Parse the XML with `ElementTree`.
   - Walk all text-bearing elements (e.g., `<hp:t>` tags inside `<hp:run>` inside `<hp:p>` paragraphs).
   - For each paragraph, collect the full concatenated text. Identify paragraphs containing old values (customer name, owner, renewal window, pricing band, escalation contact, pricing note) and replace old values with new ones in the individual `<hp:t>` elements.
   - Identify the three existing follow-up lines. Replace them with the sorted CSV follow-up items (same number of paragraphs, update text). If there are exactly three follow-up paragraphs and a different number of CSV items, add or remove paragraph elements accordingly, cloning structure from an existing follow-up paragraph.
   - **Critical: For every paragraph whose text content was modified, remove any `<hp:lineSegArray>` child element (layout cache).** This prevents overlapping-character rendering artifacts. Search for these elements using the correct namespace URI.
   - **Do NOT modify the appendix paragraph** containing `이 부록 문단은 그대로 유지해야 합니다.` — verify it remains unchanged after edits.

5. **Write back the XML.** Serialize the modified XML tree back to the same file path within the extracted directory. Use `xml_declaration=True` and `encoding='utf-8'`.

6. **Repackage the HWPX.** Create `/root/renewal_playbook_updated.hwpx` as a new ZIP file. Add every file from the extracted directory back into the ZIP, preserving the original relative paths exactly. Use `zipfile.ZIP_DEFLATED` compression. Make sure `mimetype` (if present) is the first entry and stored uncompressed (`ZIP_STORED`), as per OPC/HWPX conventions.

7. **Validate the output.**
   - Confirm `/root/renewal_playbook_updated.hwpx` exists and is a valid ZIP.
   - Re-extract and re-parse the section XML to verify:
     a. New customer name, owner, renewal window, pricing band, escalation contact, and pricing note appear.
     b. Old values do NOT appear (no duplicates).
     c. Follow-up lines match CSV items in sequence order.
     d. The appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is present and unmodified.
     e. No `<hp:lineSegArray>` elements exist in any paragraph whose text was changed.

**Important implementation notes:**
- Before parsing XML, scan the file for all `xmlns:` declarations and register each namespace prefix with `ET.register_namespace(prefix, uri)` to prevent ElementTree from renaming prefixes.
- When searching for elements, always use the full namespace URI in curly-brace notation or use a namespace map.
- Read the actual old values from the existing XML content and from the JSON to determine what to replace. Do not guess old values.
- If the JSON contains keys like `customer_name`, `current_owner`, `renewal_window`, `pricing_band`, `escalation_contact`, `pricing_note` (or similar), map each to the corresponding text in the document. The JSON may also contain the old values for reference, or you may need to infer them from the existing document.
- Handle Korean text (UTF-8) correctly throughout.

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