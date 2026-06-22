# Task Instruction

Execute the following steps to produce `/root/supplier_contact_ready.hwpx`:

1. **Inspect the workspace.** List files in the task directory to locate `supplier_contact_template.hwpx` and `supplier_contact.json`. Read `supplier_contact.json` fully to understand all key-value pairs.

2. **Unpack the HWPX template.** HWPX is a ZIP (OCF) archive. Unzip `supplier_contact_template.hwpx` into a temporary directory (e.g., `/tmp/hwpx_work/`). List the extracted contents, paying attention to `mimetype` and `Contents/section0.xml` (and any other `section*.xml` files).

3. **Read and understand the XML.** Open every `Contents/section*.xml` file. Identify:
   - All `{{...}}` placeholders (they may be split across multiple `<hp:t>` elements within a single `<hp:run>` or `<hp:p>`).
   - Korean field labels that must be preserved.
   - Any static note line.
   - The namespace URI for `hp` (expected: `urn:hancom:hwpx:1.0:hp` but detect dynamically).

4. **Write a Python script** that does the following:
   a. Parse `supplier_contact.json` into a dict.
   b. For each `section*.xml`, parse with `lxml.etree`.
   c. For each paragraph element (`<hp:p>`), collect all `<hp:t>` text nodes, concatenate them into a single string, check if the concatenated text contains any `{{...}}` pattern.
   d. If placeholders are found:
      - Perform all `{{key}}` → value substitutions using the JSON data on the concatenated string.
      - Place the fully substituted text into the **first** `<hp:t>` element of that paragraph.
      - Clear (set text to empty string or remove) any remaining `<hp:t>` elements in that paragraph so no stale fragments remain.
      - Remove any `<hp:lineSegArray>` child element from that `<hp:p>` to clear the layout cache.
   e. After processing, serialize the XML back to the file (use `xml_declaration=True, encoding='UTF-8', standalone=True` or match the original declaration).

5. **Repackage the HWPX.** Using Python's `zipfile` module:
   - Create `/root/supplier_contact_ready.hwpx` as a new ZIP.
   - Write the `mimetype` file **first**, using `ZIP_STORED` (no compression) to comply with OCF standards.
   - Write all remaining files from the unpacked directory using `ZIP_DEFLATED`.
   - Preserve the original directory structure inside the ZIP.

6. **Validate the output.**
   - Open the resulting ZIP and list its entries; confirm `mimetype` is the first entry and is stored uncompressed.
   - Read back every `section*.xml` from the ZIP and search for any remaining `{{` or `}}` patterns. Print results. There must be zero matches.
   - Confirm the file exists at `/root/supplier_contact_ready.hwpx` and is non-empty.

Key cautions:
- Placeholders in HWPX are frequently fragmented across multiple `<hp:t>` tags by the editor's internal formatting. You MUST merge text from all `<hp:t>` elements in a paragraph before doing substitution, then redistribute the result back.
- Always detect the namespace dynamically from the XML root rather than hardcoding the prefix.
- Do NOT modify paragraphs that contain no placeholders.
- Do NOT remove or alter Korean labels or the static note line.
- After editing, re-read the XML to confirm no `{{...}}` patterns survive.

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