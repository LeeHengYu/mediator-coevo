# Task Instruction

Execute the following steps to produce the event announcement HWPX document:

1. **Inspect the workspace.** List files in the task directory to locate `event_announcement_template.hwpx` and `event_data.json`. Also check for any test/verifier files (e.g., `test_output.py`, `tests/`) to understand the acceptance criteria.

2. **Read the JSON data.** Load `event_data.json` and note every key-value pair. These are the replacements for `{{key}}` placeholders.

3. **Unpack the template.** Copy `event_announcement_template.hwpx` to a working directory (e.g., `/tmp/hwpx_work/`), then unzip it there (it is a ZIP-based package). List the extracted contents.

4. **Locate placeholders.** Search all extracted XML files (especially `Contents/section0.xml`, and any `section*.xml`) for `{{` to find every placeholder. Record which files contain placeholders.

5. **Perform replacements.** For each XML file containing placeholders:
   a. Read the file as UTF-8 text.
   b. For every key in the JSON data, replace `{{key}}` with the corresponding value. Be careful with XML-special characters (`&`, `<`, `>`, `"`, `'`) — escape them properly if any JSON values contain them.
   c. After all replacements, verify no `{{` or `}}` remain in the file. If any remain, investigate whether the placeholder spans across XML tags (e.g., `{{` in one text run and `}}` in another). If so, consolidate the text runs or handle the split appropriately.

6. **Remove stale layout caches.** For every `<hp:p>` element whose text content was modified (i.e., contained a placeholder), remove any `<hp:lineSegArray>` child element (and its descendants) from that paragraph. This prevents overlapping-character rendering issues. You can do this with a regex or XML parser — just ensure the `<hp:lineSegArray>...</hp:lineSegArray>` block is fully removed from modified paragraphs.

7. **Write back the modified XML files** to their original locations within the unpacked directory.

8. **Repack the HWPX.** Re-zip the contents into `/root/event_announcement_ready.hwpx`. Important: the zip must be created from *inside* the unpacked directory so that paths like `Contents/section0.xml` are relative (no leading directory). Use `cd /tmp/hwpx_work && zip -r /root/event_announcement_ready.hwpx .` or equivalent.

9. **Validate the output.**
   a. Verify `/root/event_announcement_ready.hwpx` exists and is a valid ZIP (`unzip -t`).
   b. Extract `section0.xml` (and any other modified sections) from the output and confirm:
      - No `{{` or `}}` remain.
      - All JSON values appear in the XML text content.
      - Korean labels and static note lines are preserved.
      - No `<hp:lineSegArray>` elements exist in paragraphs that were modified.
   c. If test files exist, run `cd /root && python -m pytest test_output.py -v` (or the appropriate test command) and confirm all tests pass.

10. **If any test fails**, read the error message carefully, identify the missing or incorrect content, fix the XML, repack, and retest. Pay special attention to:
    - Exact string matching (Korean text, parenthetical annotations like the safety-audit lesson).
    - Encoding (UTF-8 throughout).
    - ZIP structure integrity.

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