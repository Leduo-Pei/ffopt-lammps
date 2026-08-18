# Security policy

## Supported versions

Only the latest tagged prerelease is actively maintained during alpha
development.

## Reporting

Do not publish credentials, private cluster paths, proprietary data, or an
exploitable security issue in a public issue. Contact the maintainer through
the email in `pyproject.toml` with a minimal reproduction and affected version.

FFOpt executes LAMMPS, MPI, Python, and generated SLURM shell scripts under the
current user's account. Review data files, plugin packages, machine
`env_setup` commands, and input paths from untrusted sources before running
them. Never store passwords, SSH keys, tokens, or API keys in `ffopt.in`,
`machines.toml`, logs, or the repository.
