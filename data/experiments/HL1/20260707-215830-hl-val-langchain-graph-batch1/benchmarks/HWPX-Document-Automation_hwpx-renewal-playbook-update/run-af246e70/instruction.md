# Task Instruction

Execute the following steps to update the HWPX renewal playbook:

1. **Inspect input files.** Read `renewal_update.json` and `followups.csv` from the task directory to understand the replacement values and follow-up sequence.

2. **Understand the HWPX structure.** The `.hwpx` file is a ZIP archive. Extract it to a temporary directory. The main content is in `Contents/section0.xml`. List all files in the archive to know what must be repacked.

3. **Parse section0.xml.** Use Python's `xml.etree.ElementTree` (with proper namespace handling for `hp:` prefixed elements). Identify all `<hp:p>` paragraphs and their `<hp:t>` text children.

4. **Apply field updates from renewal_update.json.** For each paragraph, concatenate all `<hp:t>` text into a single string. Perform string replacements for: customer name, current owner, renewal window, pricing band, escalation contact, and pricing note — replacing old values with new values everywhere they appear. After replacement, write the full result into the first `<hp:t>` element and clear (remove text from) any remaining `<hp:t>` elements in that paragraph.

5. **Replace follow-up lines.** Identify the three existing follow-up lines in the document. Replace them with the rows from `followups.csv` ordered by the `sequence` column. Each CSV row becomes one follow-up line. Remove old follow-up paragraphs that are no longer needed (if the count differs) or overwrite them in place. Ensure no duplicate or stale follow-up lines remain.

6. **Remove layout cache.** For every paragraph whose text was modified (field updates or follow-up replacement), find and remove any `<hp:lineSegArray>` child element. This prevents stale layout rendering.

7. **Preserve the appendix sentence.** Verify that the paragraph containing `이 부록 문단은 그대로 유지해야 합니다.` is unchanged — do not modify its text or remove its layout cache.

8. **Write the updated XML back** into the extracted directory at `Contents/section0.xml`.

9. **Repack the HWPX ZIP archive** to `/root/renewal_playbook_updated.hwpx`:
   - The `mimetype` file MUST be the **first entry** in the ZIP and stored with `compression=ZIP_STORED` (no compression).
   - All other files use `compression=ZIP_DEFLATED`.
   - Preserve the original directory structure exactly.

10. **Validate the output:**
    - Open the resulting `.hwpx` with `zipfile.ZipFile` and confirm it is a valid ZIP.
    - Re-read `Contents/section0.xml` from the new archive and verify:
      a. All new field values from `renewal_update.json` appear in the text.
      b. None of the old field values appear.
      c. Follow-up lines match `followups.csv` in `sequence` order.
      d. The appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is present and unmodified.
      e. No `<hp:lineSegArray>` elements exist in any modified paragraph.
    - Print confirmation of each check.

11. **Run the verifier** if a test script exists (e.g., `pytest test_output.py -v`) and confirm it passes.

IMPORTANT NOTES:
- To find old values for replacement, inspect the original `section0.xml` text and cross-reference with `renewal_update.json` which should contain both old and new values (or just new values with old values inferable from the document).
- When handling namespaces in ElementTree, register them before parsing to avoid namespace prefix mangling in output. Use `ET.register_namespace` for all namespaces found in the XML.
- When writing XML back, preserve the XML declaration and encoding.
- Do NOT use `shutil` to copy the zip; build it entry by entry to control compression per entry.

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