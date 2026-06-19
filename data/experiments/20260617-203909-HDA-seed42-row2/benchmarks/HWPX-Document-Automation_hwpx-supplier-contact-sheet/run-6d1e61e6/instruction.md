# Task Instruction

You need to update the HWPX supplier contact sheet template with values from a JSON file. Here is the step-by-step plan:

1. **Inspect the workspace**: List files in the task directory to find `supplier_contact_template.hwpx` and `supplier_contact.json`. Also check for any `test_output.py` or verifier script to understand what will be checked.

2. **Read the JSON data**: Load `supplier_contact.json` and note all key-value pairs that will be used for placeholder replacement.

3. **Examine the HWPX structure**: The `.hwpx` file is a ZIP archive. Unzip it to a temporary directory (e.g., `/tmp/hwpx_work/`) and list all files inside. The main content is typically in `Contents/section0.xml` but there may be multiple sections.

4. **Inspect all XML content files**: Read every XML file under `Contents/` (section0.xml, section1.xml, etc.) and any other text-based files in the package. Search for all `{{...}}` placeholders across ALL files in the archive, not just section0.xml. Record every placeholder found.

5. **Write and run a Python script** that does the following:
   - Reads `supplier_contact.json` into a dictionary.
   - Extracts the HWPX ZIP to a temp directory.
   - For every file in the extracted archive, if it's an XML or text file, scans for `{{...}}` patterns.
   - Replaces each `{{key}}` with the corresponding value from the JSON. Be careful with the exact placeholder format — the key inside `{{...}}` should match a JSON key (possibly after stripping whitespace).
   - **Critical**: For any `<hp:p>` paragraph element that had text modified (i.e., contained a placeholder that was replaced), remove all `<hp:lineSegArray>` child elements from that paragraph. This prevents stale layout-cache data from causing overlapping characters when the document is opened. Use an XML parser (like `lxml.etree` or `xml.etree.ElementTree`) with proper namespace handling to do this cleanly.
   - Preserve all Korean field labels and the static note line — only replace `{{...}}` patterns, nothing else.
   - Repackage the modified files back into a new ZIP file at `/root/supplier_contact_ready.hwpx`, preserving the original directory structure and using the same compression method.

6. **Validation steps** after the script runs:
   - Unzip `/root/supplier_contact_ready.hwpx` to a new temp location.
   - Search all files for any remaining `{{` patterns — there must be zero.
   - Verify the file is a valid ZIP by listing its contents.
   - Grep for a few expected JSON values in the XML to confirm they were injected.
   - Check that `<hp:lineSegArray>` elements are removed from modified paragraphs.

7. **Run the verifier**: Execute `python -m pytest test_output.py -v` (or whatever test file exists) to confirm the output passes all checks.

**Key pitfalls to avoid (from cross-task context)**:
- Make sure ALL placeholder values are correctly mapped. The failed safety-audit-brief task failed because a specific value wasn't found in the output XML. Double-check every JSON key maps to a placeholder.
- Handle XML namespaces properly when searching for and modifying `<hp:lineSegArray>` elements. The namespace URI for `hp` is typically declared in the root element.
- When repackaging the ZIP, preserve the `mimetype` file (if present) as the first entry with no compression, as HWPX follows ODF-like packaging conventions. If no mimetype file exists, just repackage normally.
- Use `re.sub` with a pattern like `\{\{\s*(.+?)\s*\}\}` to handle any whitespace inside placeholders.
- Parse XML with namespace awareness. If using ElementTree, register namespaces before parsing to avoid namespace prefix mangling in the output.

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