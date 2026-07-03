# Task Instruction

Prepare the event announcement document by following these steps:

1. **Inspect the workspace**: List files in the current directory. Read `event_data.json` to understand the replacement values. Examine the structure of `event_announcement_template.hwpx` (it is a ZIP archive).

2. **Write and run a Python script** that does the following:

   a. **Extract** `event_announcement_template.hwpx` to a temporary directory (e.g., `/tmp/hwpx_work/`).

   b. **Load `event_data.json`** and build a flat dictionary mapping each `{{key}}` placeholder string to its replacement value. If any JSON values need formatting (e.g., numbers with units, lists joined with commas, etc.), apply that formatting before building the map.

   c. **Identify all XML files** inside the extracted archive (especially files under `Contents/` such as `section0.xml`, `content.hpf`, etc.). For each XML file:
      - Read the raw XML text.
      - Register all namespaces found in the file with `xml.etree.ElementTree.register_namespace` so they are preserved on write.
      - Parse the XML with ElementTree.

   d. **Handle placeholder fragmentation**: For each `<hp:p>` (paragraph) element, collect all descendant `<hp:t>` text nodes and concatenate their `.text` values to form the full paragraph text. If the concatenated text contains any `{{...}}` placeholder:
      - Perform all placeholder replacements on the concatenated text.
      - Redistribute the replaced text back into the `<hp:t>` nodes. The simplest reliable approach: put the entire replaced text into the first `<hp:t>` node's `.text` and clear (set to empty string) the `.text` of all subsequent `<hp:t>` nodes in that paragraph.
      - **Remove layout-cache elements**: Find and remove any child element whose local tag name is `lineSegArray` (i.e., `<hp:lineSegArray>`) from that `<hp:p>` element. This prevents overlapping-character rendering artifacts.
      - Mark that this file was modified.

   e. **Write back** any modified XML files to their original paths in the extracted directory, using `ElementTree.write()` with `xml_declaration=True` and `encoding='utf-8'`.

   f. **Verify no remaining placeholders**: Re-read all XML files and assert that no `{{` or `}}` text remains anywhere.

   g. **Repackage** the extracted directory into `/root/event_announcement_ready.hwpx` using Python's `zipfile` module with `ZIP_DEFLATED` compression. Preserve the original directory structure exactly (including `[Content_Types].xml`, `META-INF/`, `Contents/`, etc.). Walk the extracted directory and add each file with its correct relative archive path.

3. **Validate the output**:
   - Confirm `/root/event_announcement_ready.hwpx` exists and is a valid ZIP.
   - List its contents to verify structural integrity.
   - Search all text content in the archive for any remaining `{{` to confirm zero leftover placeholders.
   - Spot-check that Korean labels and static note lines are preserved by grepping for a few known Korean strings from the template.

Key cautions:
- Always register XML namespaces before parsing to avoid `ns0:` prefix pollution.
- The `lineSegArray` removal must happen for EVERY paragraph that was modified, not just some.
- Do not alter paragraphs that contain no placeholders.
- Preserve all non-XML files (images, fonts, etc.) byte-for-byte by copying them as-is into the new ZIP.

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