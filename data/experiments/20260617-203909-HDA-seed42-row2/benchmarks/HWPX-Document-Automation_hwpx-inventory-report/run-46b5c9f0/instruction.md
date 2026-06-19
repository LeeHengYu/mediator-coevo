# Task Instruction

Complete the inventory status report by replacing all `{{...}}` placeholders in `inventory_report_template.hwpx` with values from `inventory_data.json`, then save the result to `/root/inventory_report_ready.hwpx`.

Step-by-step:

1. **Inspect the workspace**: List files in the task directory. Read `inventory_data.json` to understand the key-value mappings.

2. **Understand the HWPX structure**: A `.hwpx` file is a ZIP archive containing XML files (typically under `Contents/`). Unzip `inventory_report_template.hwpx` to a temporary directory (e.g., `/tmp/hwpx_work/`).

3. **Identify XML files with placeholders**: Search all extracted XML files for `{{` to find which files contain placeholders. Typically these are in `Contents/section0.xml` (and possibly other section files). List every placeholder found.

4. **Handle fragmented placeholders across XML runs**: This is the critical challenge. In HWPX XML, a single `{{placeholder}}` may be split across multiple `<hp:t>` elements within the same `<hp:run>` or across multiple `<hp:run>` elements within the same `<hp:p>` (paragraph). Strategy:
   - For each `<hp:p>` paragraph element, concatenate all text content from all `<hp:t>` children (across all `<hp:run>` elements).
   - Check if the concatenated text contains any `{{...}}` pattern.
   - If it does, perform the replacement on the concatenated text, then rewrite the paragraph's text runs: put all the replaced text into the first `<hp:run>`'s `<hp:t>` element, and clear (set to empty string) the `<hp:t>` elements in subsequent runs. Do NOT delete the `<hp:run>` elements themselves to preserve formatting structure.

5. **Remove stale layout cache**: For every paragraph (`<hp:p>`) whose text was modified, remove any `<hp:lineSegArray>` child element (and its descendants). This prevents overlapping characters when the document is opened, as the application will recalculate layout.

6. **Preserve document integrity**:
   - Keep all Korean labels and static note lines unchanged.
   - Preserve empty paragraphs (paragraphs with no text or only whitespace) exactly as they are.
   - Do not modify any paragraphs that don't contain `{{...}}` placeholders.
   - Do not alter the ZIP structure, other XML files, or binary resources.

7. **Validate before saving**:
   - Re-scan all XML content for any remaining `{{` or `}}` strings. If any remain, the replacement was incomplete — debug and fix.
   - Verify the XML is well-formed (parseable) after modifications.

8. **Repackage**: Repackage the modified files back into a ZIP archive saved as `/root/inventory_report_ready.hwpx`. Use the same compression settings. Important: when creating the ZIP, use `zipfile.ZIP_DEFLATED` and ensure the internal paths match the original archive exactly (no leading slash, same directory structure). The `mimetype` file, if present, should be stored first without compression (ZIP_STORED).

9. **Final verification**: Unzip the output file and grep for `{{` to confirm no placeholders remain. Also confirm the file is a valid ZIP.

Implementation notes:
- Use Python with `zipfile` and `xml.etree.ElementTree` (or `lxml` if available).
- Register the HWPX namespaces before parsing to avoid namespace prefix mangling. Inspect the XML to find the namespace URIs (commonly `http://www.hancom.co.kr/hwpml/2011/paragraph` for `hp:` prefix or similar). Use `ET.register_namespace()` for each namespace found.
- When writing XML back, use `ET.tostring()` with `xml_declaration=True` and `encoding='utf-8'` if the original had an XML declaration.
- Be careful with the `{{` and `}}` patterns — they may appear as `{` `{` split across runs, so the concatenation approach is essential.

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