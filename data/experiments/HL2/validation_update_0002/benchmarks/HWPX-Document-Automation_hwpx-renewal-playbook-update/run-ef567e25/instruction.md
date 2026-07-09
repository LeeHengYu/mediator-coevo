# Task Instruction

Execute the following steps to produce `/root/renewal_playbook_updated.hwpx`:

1. **Inspect the workspace.** List `/root/` and the task directory to locate `renewal_playbook.hwpx`, `renewal_update.json`, `followups.csv`, and `test_output.py`. Read `renewal_update.json` and `followups.csv` in full so you know every field and every follow-up row.

2. **Understand the source HWPX structure.** Unzip `renewal_playbook.hwpx` into a temporary directory (e.g., `/tmp/hwpx_src`). List the extracted tree. Read `Contents/section0.xml` carefully — this is the main editable content. Also note the `mimetype` file content (should be a single line, no newline).

3. **Register all XML namespaces before parsing.** Before calling `ET.parse()` or `ET.fromstring()`, register every namespace declared in the root element of `section0.xml` using `xml.etree.ElementTree.register_namespace(prefix, uri)`. This prevents ElementTree from rewriting prefixes to `ns0:`, `ns1:`, etc. Common HWPX namespaces include:
   - `hp` → `http://www.hancom.co.kr/hwpml/2011/paragraph`
   - `hp10` → `http://www.hancom.co.kr/hwpml/2016/paragraph`  (if present)
   - `hs` → `http://www.hancom.co.kr/hwpml/2011/section`
   - and others — read them from the file itself.

4. **Parse section0.xml and perform text replacements.** Walk every element. For each text node (element.text or element.tail) that contains an old value from the JSON, replace it with the new value. The JSON will specify pairs like old customer name → new customer name, old owner → new owner, old renewal window → new, old pricing band → new, old escalation contact → new, old pricing note → new. Replace **everywhere** these appear.

5. **Replace the three follow-up lines.** Identify the three existing follow-up paragraphs in the XML (they will be consecutive paragraphs whose text content matches a pattern like "Day X: …" or numbered follow-up items). Read `followups.csv`, sort rows by the `sequence` column, and replace the text of those three paragraphs with the CSV items in sequence order. If the follow-up text is spread across multiple `<hp:run>/<hp:t>` child elements within a paragraph, consolidate into a single `<hp:t>` element per paragraph (or replace the text of the first `<hp:t>` and remove extras) so no stale/duplicate content remains.

6. **Remove stale layout caches.** For every paragraph (`<hp:p>`) whose text you modified in steps 4 or 5, find and remove all `<hp:lineSegArray>` child elements (and their children). This prevents overlapping-character rendering in HWP viewers.

7. **Preserve the appendix sentence.** Verify that the sentence `이 부록 문단은 그대로 유지해야 합니다.` still exists unchanged in the XML after all edits. Do not modify the paragraph containing it.

8. **Serialize the modified XML.** Write the modified ElementTree back to `Contents/section0.xml` in the temp directory, using `xml_declaration=True, encoding='UTF-8'`. Confirm the written file is well-formed XML.

9. **Repackage as a valid HWPX (OCF/ZIP).** Create `/root/renewal_playbook_updated.hwpx` as a ZIP file:
   - First entry: `mimetype` — stored (compression=ZIP_STORED, no extra field) with the exact original content.
   - Then add every other file from the extracted tree (including the modified `Contents/section0.xml`) using ZIP_DEFLATED.
   - Preserve the original directory structure.

10. **Validate.** Unzip `/root/renewal_playbook_updated.hwpx` to a new temp dir and:
    - Confirm `mimetype` is the first entry and is uncompressed.
    - Parse `Contents/section0.xml` and verify:
      a. New customer name, owner, renewal window, pricing band, escalation contact, and pricing note appear.
      b. Old values do NOT appear.
      c. The three follow-up lines match the CSV rows in sequence order.
      d. The appendix sentence is intact.
      e. No `<hp:lineSegArray>` elements exist in modified paragraphs.

11. **Run the verifier.** Execute `cd /root && python -m pytest test_output.py -v` (or wherever the test file is). Confirm all tests pass.

Write the solution as a single Python script and execute it. If any step fails, diagnose, fix, and re-run.

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