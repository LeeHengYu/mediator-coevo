# Task Instruction

Execute the following steps in order:

1. **Inspect the workspace.** List the files in the current working directory to locate `event_announcement_template.hwpx` and `event_data.json`. Also check if there is a `test_output.py` or any verifier script.

2. **Read the JSON data.** Print the contents of `event_data.json` so you know every key-value pair that must be substituted.

3. **Examine the HWPX template.** A `.hwpx` file is a ZIP archive. List its entries with `python3 -c "import zipfile; z=zipfile.ZipFile('event_announcement_template.hwpx'); z.printdir()"`. Identify all XML files inside (typically under `Contents/`). For each XML entry, print its contents and search for `{{` to locate every placeholder. Pay special attention to `section0.xml` (or similarly named content XML) which holds the document body.

4. **Write a single Python script (`fill_template.py`)** that does the following:

   a. Opens the template HWPX as a ZIP, copies it to `/root/event_announcement_ready.hwpx` as a new ZIP, processing each member:
      - For every XML member, decode it as UTF-8 and perform placeholder replacement (step b–e below), then write the result.
      - For non-XML members, copy bytes unchanged.

   b. **Parse each XML file with `xml.etree.ElementTree`.** For every `<hp:p>` (paragraph) element (namespace may vary — detect the actual namespace from the file), collect all text runs (`<hp:t>` or equivalent) in document order.

   c. **Concatenate the text content of all runs in a paragraph** into a single string. This is critical because placeholders like `{{event_name}}` may be split across multiple `<hp:t>` elements.

   d. **Perform replacements.** For every key in `event_data.json`, replace `{{key}}` with the corresponding value in the concatenated string. After all replacements, confirm no `{{…}}` pattern remains in that paragraph's text.

   e. **If any replacement was made in a paragraph:**
      - Set the first `<hp:t>` element's text to the fully-replaced string.
      - Remove all subsequent `<hp:t>` elements (and their parent `<hp:run>` if the run becomes empty).
      - **Remove all `<hp:lineSegArray>` (or `<lineSegArray>` in whatever namespace) child elements** from that paragraph to clear the layout cache. Search for any element whose local tag name is `lineSegArray` regardless of namespace prefix.

   f. **Re-serialize** the modified XML tree back to a UTF-8 string (with XML declaration) and write it into the output ZIP.

   g. After writing the ZIP, reopen it and scan every XML entry for any remaining `{{` to verify no placeholders survive. Print a confirmation or raise an error.

5. **Run the script:** `python3 fill_template.py`

6. **Validate the output:**
   - Confirm `/root/event_announcement_ready.hwpx` exists and is a valid ZIP: `python3 -c "import zipfile; print(zipfile.is_zipfile('/root/event_announcement_ready.hwpx'))"`
   - List its entries.
   - Print the content XML(s) and visually confirm all placeholders are replaced, Korean labels are intact, and no `lineSegArray` remains in modified paragraphs.

7. **Run the verifier** if `test_output.py` exists: `cd /root && python3 -m pytest test_output.py -v`

Key pitfalls to avoid:
- **Do NOT forget to actually write the output file.** (A prior sibling task failed because the output file was never created.)
- Handle namespace prefixes carefully. Use `ElementTree` namespace-aware parsing. Register namespaces before serialization so they are preserved.
- When removing elements, iterate over a copy of the list to avoid mutation-during-iteration bugs.
- Preserve the static note line and all Korean text — only `{{…}}` tokens should change.
- If the template has multiple content XML files, process ALL of them, not just `section0.xml`.

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