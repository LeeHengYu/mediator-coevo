# Task Instruction

Complete the inventory status report by following these steps:

1. **Inspect the workspace**: List files in the current directory to find `inventory_report_template.hwpx` and `inventory_data.json`.

2. **Read the JSON data**: `cat inventory_data.json` to understand all key-value pairs that will replace placeholders.

3. **Unzip the HWPX template**: Create a working directory and unzip the template:
   ```
   mkdir -p /root/hwpx_work
   cd /root/hwpx_work
   unzip -o /root/inventory_report_template.hwpx
   ```
   (Adjust the source path if the template is located elsewhere.)

4. **Identify XML files with placeholders**: Search for `{{` across all XML files in the extracted archive:
   ```
   grep -rl '{{' /root/hwpx_work/
   ```
   Focus on `Contents/section0.xml` (and any other section files found).

5. **Read the section XML carefully**: `cat` the section XML file(s) to understand the structure. Pay attention to:
   - Placeholders like `{{placeholder_name}}` that may be split across multiple XML elements (e.g., `{{` in one `<hp:t>` and `}}` in another).
   - Korean text and empty paragraphs that must be preserved.
   - `<hp:linesegarray>` elements that serve as layout cache.

6. **Write a Python script** to perform the replacements. The script should:
   a. Load `inventory_data.json`.
   b. Read the raw XML content of each section file.
   c. **First**, collapse split placeholders: remove XML tags between `{{` and `}}` so that each placeholder becomes a single continuous `{{key}}` string within one text run. A robust approach: use a regex to find `\{\{[^}]*\}\}` after stripping inner XML tags, or iteratively join adjacent `<hp:t>` elements that together form a placeholder.
   d. **Then**, replace each `{{key}}` with the corresponding value from the JSON data. Ensure all values are converted to strings.
   e. **Remove `<hp:linesegarray>` elements** from any `<hp:p>` paragraph that had a placeholder replaced. This is critical — the verifier checks that modified paragraphs do not retain stale layout cache. Use an XML parser (e.g., `lxml` or `xml.etree.ElementTree`) or a regex like `<hp:linesegarray[^>]*>.*?</hp:linesegarray>` (with `re.DOTALL`) to strip these elements from modified paragraphs.
   f. Verify no `{{` remains in the output XML.
   g. Write the modified XML back to the file.

7. **Verify the result**:
   - `grep -c '{{' /root/hwpx_work/Contents/section0.xml` should return 0.
   - Spot-check that Korean labels and empty paragraphs are intact.
   - Check that no `<hp:linesegarray>` exists in paragraphs that were modified.

8. **Repack the HWPX archive**: From the working directory, create the output file:
   ```
   cd /root/hwpx_work
   zip -r /root/inventory_report_ready.hwpx . -x '.*'
   ```
   Ensure the zip is created from the root of the extracted content (so paths like `Contents/section0.xml` are correct relative to the archive root, matching the original structure).

9. **Final validation**:
   - `unzip -l /root/inventory_report_ready.hwpx` to confirm the archive structure matches the original.
   - `python3 -c "import zipfile; z=zipfile.ZipFile('/root/inventory_report_ready.hwpx'); print(z.namelist())"` to verify it's a valid zip.
   - Grep the output archive's section XML for any remaining `{{` placeholders.

**Key pitfalls to avoid (from cross-task feedback)**:
- Do NOT skip the layout cache removal step. The verifier explicitly checks that modified paragraphs have no `<hp:linesegarray>` elements. Remove them from every paragraph where text was changed.
- Handle placeholders that are split across multiple XML inline elements by joining/collapsing them before replacement.
- Preserve all non-placeholder content exactly: Korean text, static notes, empty paragraphs.

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