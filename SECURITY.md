# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please report (suspected) security vulnerabilities to **[kyletjones@gmail.com](mailto:kyletjones@gmail.com)**. You will receive a response within 48 hours. If the issue is confirmed, we will release a patch as soon as possible depending on complexity but historically within a few days.

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

This information will help us triage your report more quickly.

## Security Best Practices

PlotSmith is a plotting library that processes user-provided data. To use it securely:

1. **Validate Input Data**: Always validate and sanitize data before passing it to PlotSmith functions
2. **File Paths**: When using `save_path`, ensure paths are validated to prevent directory traversal attacks
3. **Dependencies**: Keep dependencies up to date to receive security patches
4. **Environment**: Run PlotSmith in a secure environment with appropriate permissions

## Disclosure Policy

When the security team receives a security bug report, they will assign it to a primary handler. This person will coordinate the fix and release process, involving the following steps:

1. Confirm the problem and determine the affected versions
2. Audit code to find any similar problems
3. Prepare fixes for all releases still under maintenance
4. Publish a security advisory

We credit security researchers who report vulnerabilities responsibly.

