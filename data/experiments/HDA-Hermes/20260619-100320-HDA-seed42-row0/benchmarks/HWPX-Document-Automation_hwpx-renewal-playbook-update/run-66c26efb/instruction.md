# Task Instruction

Create and run a Python script that updates the HWPX renewal playbook. Follow these steps precisely:

1. **Inspect inputs**: Read `renewal_update.json` and `followups.csv` from the task directory to understand the replacement values and follow-up items.

2. **Extract the HWPX**: The file `renewal_playbook.hwpx` is a ZIP archive. Extract it to a temporary directory.

3. **Identify the content XML files**: List all XML files inside the extracted archive (typically under `Contents/section*.xml` or similar). Inspect them to find where editable text paragraphs live.

4. **Parse and modify using ElementTree (not regex for structure)**:
   - Use `xml.etree.ElementTree` with proper namespace handling. Register the `hp` namespace (and any others found in the XML declaration) so output preserves namespace prefixes.
   - For each section XML file:
     a. **Consolidate split `<hp:t>` tags**: Within each `<hp:run>` element, if there are multiple `<hp:t>` child elements, merge their `.text` content into the first `<hp:t>` and remove the rest. This prevents placeholders or values from being split across tags.
     b. **Perform text replacements**: Using the values from `renewal_update.json`, replace the customer name, current owner, renewal window, pricing band, escalation contact, and pricing note everywhere they appear in `<hp:t>` text content. You need to identify the OLD values by reading the original XML first, then replace with the NEW values from the JSON.
     c. **Replace follow-up lines**: Read `followups.csv`, sort rows by the `sequence` column (ascending), and replace the three existing follow-up lines with the CSV items in sequence order. Identify follow-up lines by their content pattern (e.g., numbered items like `1.`, `2.`, `3.` or similar pattern in the original).
     d. **Preserve the appendix sentence**: Ensure the text `이 부록 문단은 그대로 유지해야 합니다.` is never modified.
     e. **Remove `<hp:lineSegArray>` from modified paragraphs**: For every `<hp:p>` paragraph element that was modified (text changed), find and remove ALL `<hp:lineSegArray>` child elements using the ElementTree API (not regex). Use `element.findall()` with the proper namespace and `element.remove()` to guarantee complete removal. After removal, verify with another `findall()` that none remain.

5. **Write back the modified XML**: Serialize each modified XML tree back to the file, preserving the XML declaration and encoding.

6. **Re-zip as HWPX**:
   - Create the output ZIP at `/root/renewal_playbook_updated.hwpx`.
   - Add the `mimetype` file FIRST, using `ZIP_STORED` (no compression, compression level 0).
   - Add all other files using `ZIP_DEFLATED`.
   - Preserve the original directory structure exactly.

7. **Validate**:
   - Open the output `.hwpx` as a ZIP and verify it's valid.
   - Re-parse the section XML from the output ZIP and confirm:
     - The new values from `renewal_update.json` appear in the text.
     - The follow-up items from `followups.csv` appear in sequence order.
     - The appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is present and unchanged.
     - No `<hp:lineSegArray>` elements exist in any paragraph that contains modified text.
     - Old values do not appear (no duplicates).

IMPORTANT NOTES:
- Use ElementTree for ALL XML manipulation, not regex. The cross-task feedback shows regex-based `<hp:lineSegArray>` removal has failed in similar tasks.
- When consolidating `<hp:t>` tags, work within `<hp:run>` elements to preserve formatting structure.
- To identify old values for replacement: first scan the original XML to extract current field values, then map them to the new values from the JSON update file.
- Track which `<hp:p>` elements are modified so you can surgically remove their `<hp:lineSegArray>` children.

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