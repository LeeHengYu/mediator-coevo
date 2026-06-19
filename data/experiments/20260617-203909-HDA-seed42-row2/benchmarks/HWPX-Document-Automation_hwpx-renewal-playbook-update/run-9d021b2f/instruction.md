# Task Instruction

You must revise the existing HWPX renewal playbook and save the updated version. Follow these steps precisely:

1. **Inspect the workspace.** List files in the task directory to locate `renewal_playbook.hwpx`, `renewal_update.json`, and `followups.csv`. Read `renewal_update.json` and `followups.csv` to understand the update values and follow-up items.

2. **Understand the HWPX structure.** A `.hwpx` file is a ZIP archive containing XML files (typically under `Contents/`). Unzip `renewal_playbook.hwpx` into a temporary directory (e.g., `/tmp/hwpx_work/`) to inspect its structure. List all files in the archive. Identify the section XML files (e.g., `section0.xml`, `section1.xml`, etc.) under `Contents/`.

3. **Read and parse the update data:**
   - Parse `renewal_update.json` to get the new values for: customer name, current owner, renewal window, pricing band, escalation contact, and pricing note.
   - Parse `followups.csv` to get the follow-up items. Sort them by the `sequence` column to determine the correct order.

4. **Read each section XML file** and identify:
   - All occurrences of the OLD values for customer name, current owner, renewal window, pricing band, escalation contact, and pricing note. You'll need to find the current/old values by reading the XML content first.
   - The three existing follow-up lines that need to be replaced.
   - The appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` — this must NOT be changed.

5. **Perform text replacements** in the section XML files:
   - Replace every occurrence of each old field value with the corresponding new value from the JSON.
   - Replace the three follow-up lines with the CSV items in `sequence` order. Make sure you remove the old follow-up text entirely (no duplicates).
   - Do NOT modify the appendix sentence.

6. **Critical: Remove layout-cache elements.** For every `<hp:p>` paragraph element whose text content you modified, remove any `<hp:lineSegArray>` child element (and its entire subtree). This is essential — stale layout caches cause overlapping characters when the document is opened. Use an XML parser (e.g., Python's `lxml` or `xml.etree.ElementTree`) for reliable manipulation rather than regex.

7. **Reassemble the HWPX package.** Repackage the modified files back into a valid ZIP archive saved as `/root/renewal_playbook_updated.hwpx`. Preserve the original ZIP structure exactly (same directory paths, same file entries). Use `zipfile.ZipFile` in Python with appropriate compression.

8. **Validate the output:**
   - Verify `/root/renewal_playbook_updated.hwpx` exists and is a valid ZIP.
   - Unzip it and check that the section XMLs contain the new values from the JSON.
   - Confirm the follow-up items appear in the correct sequence order.
   - Confirm the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is present and unchanged.
   - Confirm no `<hp:lineSegArray>` elements remain in any paragraph whose text was modified.
   - Confirm old values do not appear anywhere in the editable sections.

Use Python for all XML parsing and ZIP manipulation. Prefer `lxml` if available, otherwise `xml.etree.ElementTree`. Be careful with XML namespaces — inspect the actual namespace URIs in the files before writing XPath queries.

If the verifier test file `test_output.py` exists, run `cd /root && python -m pytest test_output.py -v` at the end to confirm the result passes.

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