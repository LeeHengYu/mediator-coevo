# Task Instruction

Prepare the event announcement document by replacing all `{{...}}` placeholders with values from the JSON data file, then save the result as a valid `.hwpx` package.

## Steps

1. **Inspect the workspace.** List files in the current directory and `/root/` to locate `event_announcement_template.hwpx` and `event_data.json`.

2. **Read the JSON data.** Load `event_data.json` and print its contents so you know every key-value pair available for substitution.

3. **Unzip the HWPX template.** HWPX files are ZIP archives. Unzip `event_announcement_template.hwpx` into a temporary working directory (e.g., `/tmp/hwpx_work/`).

4. **List all files inside the extracted archive.** Identify every XML file (especially files under `Contents/` such as `section0.xml`, `content.hpf`, etc.).

5. **Process each XML file with `lxml`.** For every `.xml` file in the extracted archive:
   a. Parse it with `lxml.etree` (use `recover=True` if needed).
   b. Collect all namespaces from the root element dynamically.
   c. Find all `<hp:p>` paragraph elements (or equivalent, respecting the actual namespace URI for the `hp` prefix).
   d. For each paragraph:
      - Gather all `<hp:t>` text-run elements.
      - Concatenate all their `.text` content into a single string to reconstruct the full paragraph text (placeholders may be split across multiple `<hp:t>` elements).
      - Check whether the concatenated text contains any `{{...}}` placeholder pattern.
      - If it does:
        1. Replace every `{{key}}` with the corresponding value from `event_data.json`. Match keys case-sensitively.
        2. Set the first `<hp:t>` element's `.text` to the fully-substituted string.
        3. Remove all subsequent `<hp:t>` elements in that paragraph (clear stale split runs).
        4. **Remove any `<hp:lineSegArray>` element** (layout cache) within that paragraph to prevent overlapping-character rendering artifacts.
   e. Write the modified XML back to the same file path, using `xml_declaration=True`, `encoding='UTF-8'`, and `standalone=True` (or match the original declaration).

6. **Verify no placeholders remain.** After processing, scan every XML file in the working directory for any remaining `{{` or `}}` strings. Print a confirmation or list any residual placeholders.

7. **Re-pack the HWPX file.** Re-zip the contents of the working directory into `/root/event_announcement_ready.hwpx`. Use `zipfile.ZipFile` with `ZIP_DEFLATED` compression. **Important:** The zip must preserve the original directory structure exactly (e.g., `Contents/section0.xml`, `META-INF/...`, etc.). Do NOT include the working directory root itself as a prefix—archive paths should be relative from inside the extracted folder.

8. **Validate the output.**
   - Confirm `/root/event_announcement_ready.hwpx` exists and is a valid ZIP.
   - Open it with `zipfile.ZipFile` and list its entries to verify structure matches the original template.
   - Read back the main content XML (e.g., `Contents/section0.xml`) and confirm no `{{...}}` patterns remain.
   - Print a summary: number of placeholders replaced, files modified, and final file size.

## Key Cautions
- **Namespace handling:** The `hp` prefix namespace URI varies across HWPX versions. Extract it dynamically from the root element's `nsmap` rather than hardcoding.
- **Layout cache removal:** Always remove `<hp:lineSegArray>` (and its children) from any paragraph you modify. This is critical for clean rendering.
- **Korean text preservation:** Do not alter any text that doesn't contain `{{...}}` placeholders. Korean labels and static note lines must remain byte-identical.
- **Text run consolidation:** Placeholders like `{{event_name}}` are often split across multiple `<hp:t>` elements (e.g., `{{event`, `_name`, `}}`). Always concatenate all runs in a paragraph before doing regex replacement.
- **ZIP structure:** The HWPX format requires the archive structure to be exact. Do not add extra directories or change file paths.

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