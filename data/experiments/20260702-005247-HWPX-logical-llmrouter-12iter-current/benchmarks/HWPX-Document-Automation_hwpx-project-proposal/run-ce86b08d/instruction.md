# Task Instruction

Complete the project proposal document by following these steps:

1. **Inspect the workspace:**
   ```
   ls /root/
   ```
   Locate `project_proposal_template.hwpx` and `project_proposal.json`.

2. **Read the JSON data:**
   ```
   cat /root/project_proposal.json
   ```
   Note all key-value pairs. For any budget/currency value, plan to strip commas while keeping the leading currency symbol (e.g., `₩1,500,000,000` → `₩1500000000`).

3. **Unzip the HWPX template into a working directory:**
   ```
   mkdir -p /root/hwpx_work
   cd /root/hwpx_work
   unzip /root/project_proposal_template.hwpx -d template_extracted
   ```

4. **Explore the extracted structure:**
   ```
   find /root/hwpx_work/template_extracted -type f
   ```
   Identify all XML section files (e.g., `Contents/section0.xml`, `Contents/section1.xml`, etc.).

5. **Inspect each section XML file** to understand:
   - Where `{{...}}` placeholders appear
   - Whether placeholders are split across multiple XML tags (e.g., `{{` in one `<hp:t>` and `}}` in another)
   - Where phase lines (단계1, 단계2, 단계3) with date ranges appear
   - Where `<hp:linesegarray>` layout cache elements exist

6. **Write and run a Python script** (`/root/hwpx_work/process.py`) that does the following:

   a. **Load the JSON** from `/root/project_proposal.json`.

   b. **For each section XML file** in the extracted template:
      - Parse with `xml.etree.ElementTree`, preserving namespaces.
      - **Register all namespaces** before parsing to avoid `ns0:` prefix pollution. Read the file first to extract namespace declarations, then register them with `ET.register_namespace()`.

   c. **Handle split placeholders:** For each `<hp:p>` paragraph element, concatenate all `<hp:t>` text content to find complete `{{key}}` patterns. If a placeholder spans multiple `<hp:t>` tags, merge the text into a single `<hp:t>` and remove the now-empty tags. Then perform the replacement on the merged text.

   d. **Replace all `{{key}}` placeholders** with corresponding JSON values. For budget values containing commas with a currency symbol, strip commas but keep the symbol.

   e. **Append month spans to phase lines:** After placeholder replacement, for any paragraph text containing `단계1`, `단계2`, or `단계3` followed by a date range, calculate the month span between the two dates and append ` (N개월)` to the paragraph text. The date range format will be something like `2025.01 ~ 2025.03`; calculate months as `(end_year - start_year) * 12 + (end_month - start_month)` or by counting inclusive months as appropriate. Expected results: 단계1 → (3개월), 단계2 → (3개월), 단계3 → (1개월).

   f. **Remove `<hp:linesegarray>` elements** from any `<hp:p>` paragraph that was modified (had text changed). This is critical — stale layout cache causes rendering issues and test failures.

   g. **Verify no `{{` remains** in any text node of the output XML. If any remain, log an error.

   h. **Write the modified XML** back to the same file path in the extracted directory.

   i. **Re-zip** the extracted directory into `/root/project_proposal_ready.hwpx`, preserving the original directory structure (files at the root of the zip, not nested under an extra folder). Use `zipfile.ZipFile` with proper relative paths.

7. **Run the script:**
   ```
   cd /root/hwpx_work && python3 process.py
   ```

8. **Validate the output:**
   - Unzip `/root/project_proposal_ready.hwpx` to a temp directory and inspect the section XML files.
   - Confirm no `{{` text remains anywhere.
   - Confirm phase lines have the `(N개월)` suffix.
   - Confirm budget values have no commas but retain the currency symbol.
   - Confirm modified paragraphs do NOT contain `<hp:linesegarray>` elements.
   - Confirm unmodified paragraphs still retain their original structure.
   - Confirm the file is a valid zip.

**Key pitfalls to avoid:**
- Placeholders may be split across multiple `<hp:t>` elements within a single `<hp:run>` or across multiple `<hp:run>` elements. You MUST concatenate and merge before replacing.
- Namespace handling: register all namespaces from the original XML before writing to avoid prefix renaming.
- When re-zipping, walk from inside the extracted directory so paths are relative (no leading folder name).
- The `linesegarray` removal must happen for ALL modified paragraphs, not just some.

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