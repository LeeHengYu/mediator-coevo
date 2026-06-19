# Task Instruction

Complete the inventory status report by filling in the HWPX template with JSON data.

Steps:

1. Read `inventory_data.json` to understand the data structure and all available keys/values.
2. Inspect the HWPX template `inventory_report_template.hwpx` — it is a ZIP archive. List its contents and read the main content XML file (typically `Contents/section0.xml` or similar) to find all `{{...}}` placeholders.
3. Flatten the JSON data if it contains nested objects, using dot-notation keys (e.g., if the JSON has `{"warehouse": {"name": "X"}}`, create a flat mapping like `warehouse.name` → `X`). Also keep top-level keys as-is. Build a comprehensive placeholder-to-value mapping that covers every `{{...}}` pattern found in the XML.
4. Perform an in-flight ZIP copy from the template to `/root/inventory_report_ready.hwpx`:
   - For each entry in the source ZIP, copy it unchanged to the output ZIP, EXCEPT for XML files that contain `{{` placeholders (likely `Contents/section0.xml` or similar content XML).
   - For those XML files:
     a. Decode the XML content.
     b. Replace every `{{...}}` placeholder with the corresponding value from the flattened JSON mapping. Handle type conversion: numbers should be inserted as their string representation.
     c. After performing replacements, strip all `<hp:linesegarray>` elements (including their children and closing tags) from any paragraph (`<hp:p>`) whose text content was modified by a replacement. This prevents stale layout-cache data from causing overlapping characters when the document is opened. Use regex or XML parsing to remove these elements.
     d. Verify no `{{` remains in the processed XML.
     e. Write the modified XML back to the output ZIP.
5. After creating the output file, validate:
   - `/root/inventory_report_ready.hwpx` exists and is a valid ZIP.
   - Open it and read the content XML; confirm zero `{{` placeholders remain.
   - Confirm Korean labels and static note lines are preserved.
   - Confirm empty paragraphs are still present in the document structure.
   - Print a summary of replacements made and verification results.

Use Python with the `zipfile` module and `re` for regex operations. Do not install external packages.

IMPORTANT: The placeholder keys in the template may use various formats (dot notation, underscores, nested paths). Carefully inspect the actual placeholder strings in the XML and match them exactly against the JSON data keys. If placeholders use a format like `{{warehouse_name}}`, map from the JSON accordingly. Build the mapping by inspecting both the JSON structure AND the actual placeholder strings found in the XML.

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