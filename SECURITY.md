# Security Policy

This policy covers `langgraph-checkpoint-redis`, maintained in
`redis-developer/langgraph-redis`, including its Redis-backed checkpoint
savers, stores, serialization, and agent middleware.

## Reporting a Vulnerability

Please report suspected security vulnerabilities privately. Do not include
vulnerability details or reproduction code in public issues, discussions,
or pull requests.

You can report a vulnerability through either of these channels:

- **GitHub private vulnerability reporting:** Open the repository’s
  [Security tab](https://github.com/redis-developer/langgraph-redis/security)
  and select **Report a vulnerability** to share details privately with
  the maintainers.
- **Redis Vulnerability Disclosure Program:** Submit a report through the
  [Redis disclosure program](https://redis.io/redis-responsible-vulnerability-disclosure/),
  identifying the repository as
  `https://github.com/redis-developer/langgraph-redis` and the Python
  package as `langgraph-checkpoint-redis`.

For questions about reporting or difficulty accessing either channel,
contact security@redis.com.

### What to Include

Please provide as much of the following as possible:

- A description of the vulnerability and its potential security impact.
- The affected package version or commit.
- A minimal reproduction, including code and required configuration.
- Python, Redis, `redis`, `redisvl`, and `langgraph-checkpoint` versions.
- The affected component, including whether it uses synchronous,
  asynchronous, full, or shallow checkpointing.
- Relevant deployment details, such as standalone Redis, Redis Cluster,
  or a managed Redis service.
- The access an attacker needs and which inputs they can control.
- Any proposed mitigation or fix, if available.

For issues involving checkpoints, stores, or middleware, explain how
thread identifiers, namespaces, cache configuration, or stored data
contribute to the issue.

Use synthetic data and remove credentials, connection secrets, personal
information, and customer data from reproductions.

You do not need to establish the root cause before reporting.

## Handling Reports

Reports are assessed with Redis Product Security and the project
maintainers. Assessment includes validating the reproduction, identifying
affected versions and dependencies, and determining impact and severity.

For confirmed vulnerabilities, we coordinate remediation and disclosure
with the reporter. We may request additional information or help
validating a fix. Response and remediation times depend on the severity
and complexity of the issue.

Please coordinate public disclosure with us so that users can receive
fixes or mitigations before technical details are published.

## Supported Versions and Security Fixes

Security fixes target the latest released version of
`langgraph-checkpoint-redis`. Users should upgrade to the latest release
and review its dependency requirements.

Backports to older releases are considered case by case, based on
severity, compatibility, and maintenance effort. Maintenance of every
previous release is not guaranteed.

When a security fix is released, the advisory or release notes will
identify affected versions, fixed versions, and any available mitigations.
Some fixes may require upgrading dependencies.

## Scope and Deployment Context

We welcome reports involving the confidentiality, integrity, or
availability of checkpoint data, stored memory, cached responses, and
tool results. Examples include unsafe deserialization, unintended data
access or modification, and query or key handling that accesses data
outside the requested scope.

The library runs within an application and uses that application's Redis
credentials. Applications are responsible for authenticating callers,
authorizing access to threads and namespaces, and configuring appropriate
Redis access controls, network protection, and TLS.

When reporting an issue, describe these application controls and the
trust boundary you believe the library crosses. If you are unsure whether
an issue belongs to this library or a dependency, report it privately and
include the relevant details.
