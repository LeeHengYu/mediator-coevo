# Task Instruction

Prepare the event announcement document by replacing all `{{...}}` placeholders with values from `event_data.json`, then save the result to `/root/event_announcement_ready.hwpx`.

## Steps

### 1. Inspect the input files
- Read `event_data.json` to understand the keys and values available.
- List the contents of `event_announcement_template.hwpx` (it's a ZIP archive) to identify all XML files inside.
- Extract the HWPX to a temporary working directory (e.g., `/tmp/hwpx_work`).

### 2. Identify placeholders
- Search all extracted XML files (especially under `Contents/`) for any occurrence of `{{` to locate every placeholder.
- Note which files and which XML elements contain placeholders.

### 3. Replace placeholders using Python
Write and run a Python script that does the following:

#### a. Handle placeholder fragmentation
HWPX XML often splits `{{placeholder}}` across multiple `<hp:t>` text nodes within a paragraph's runs. To handle this:
- For each paragraph element (`<hp:p>`), concatenate all `<hp:t>` text content.
- Check if the concatenated text contains any `{{...}}` pattern.
- If it does, perform all replacements on the concatenated string using the JSON data.
- Then place the fully-replaced text into the **first** `<hp:t>` node of that paragraph and clear (or remove) the remaining `<hp:t>` nodes' text, so the replacement is clean.

#### b. Remove layout-cache elements from modified paragraphs
For every paragraph (`<hp:p>`) whose text was modified:
- Find and remove any `<hp:lineSegArray>` child elements (and any other layout-cache elements like `<hp:lineseg>`).
- This ensures the word processor recalculates character positions and prevents overlapping text.

#### c. Register all XML namespaces properly
Before parsing, register all namespaces found in the XML files (especially `hp`, `hp10`, etc.) using `xml.etree.ElementTree.register_namespace()` so they are preserved on output. Read the root element's namespace declarations to discover them.

#### d. Preserve Korean labels and static content
Only modify text that contains `{{...}}` placeholders. Do not alter any other text content.

### 4. Repackage the HWPX
- After modifying the XML files in place within the extracted directory, repackage everything back into a ZIP file saved as `/root/event_announcement_ready.hwpx`.
- Use `zipfile.ZipFile` with `ZIP_DEFLATED` compression.
- Ensure all original files (including non-XML files like images, mimetype, etc.) are included with their original directory structure.

### 5. Validate the output
- Open the resulting `/root/event_announcement_ready.hwpx` as a ZIP and verify it's valid.
- Search all XML content inside the output HWPX for any remaining `{{` — there must be **zero** occurrences.
- Print a summary of replacements made and confirmation that no placeholders remain.

## Key Pitfalls to Avoid
- **Fragmented placeholders**: Always work at the paragraph level, not individual text-node level.
- **Layout cache**: Always strip `<hp:lineSegArray>` (and similar cache elements) from any paragraph you modify.
- **Namespace preservation**: Register all namespaces before parsing to avoid `ns0:` prefix pollution in output.
- **ZIP integrity**: Include every file from the original archive, not just the XMLs you modified.

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