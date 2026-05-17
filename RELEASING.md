# Releasing pyds4

The first PyPI release should be `0.1.0`. The package is still marked alpha,
and that version is already the project metadata version.

## One-time PyPI setup

Create a PyPI pending trusted publisher before the first GitHub Actions
release:

- PyPI project name: `pyds4`
- Owner: `avalan-ai`
- Repository: `pyds4`
- Workflow name: `release.yml`
- Environment: `pypi`

Also create a GitHub environment named `pypi`. Use required reviewers there if
you want a manual gate before publication.

## Local release

Local publishing uses Twine credentials, not GitHub OIDC. For the first local
upload of a new PyPI project, use an account-scoped PyPI token. Subsequent
local uploads can use a project-scoped token.

```sh
make release-tools
DS4_SOURCE_DIR=/path/to/ds4 PYDS4_BACKEND=metal make release
```

The DS4 checkout must be at the pinned commit in `Makefile` and
`CMakeLists.txt`. The release target refuses to build a wheel without
`DS4_SOURCE_DIR`, so it cannot accidentally publish an import-only native
extension.

To publish to TestPyPI instead:

```sh
DS4_SOURCE_DIR=/path/to/ds4 PYDS4_BACKEND=metal \
  make release TWINE_UPLOAD_ARGS="--repository-url https://test.pypi.org/legacy/"
```

## GitHub Actions release

Run the `Release` workflow manually with the version to publish. The workflow
checks that `pyproject.toml` already contains that version, builds the sdist
and backend wheels for Python 3.11 and 3.12, publishes through PyPI trusted
publishing, tags the commit, and creates a GitHub release.

The workflow builds:

- macOS arm64 Metal wheels on `macos-15`.
- Linux x86_64 CUDA wheels in `nvidia/cuda:12.6.3-devel-rockylinux8`.

The `cuda_arch` workflow input is passed to `CMAKE_CUDA_ARCHITECTURES` through
`CUDA_ARCH`; the default is `90`.

CUDA wheels are audited with `auditwheel show`, but they intentionally keep
CUDA runtime and cuBLAS libraries external and are uploaded as `linux_x86_64`
wheels. A local `auditwheel repair` test bundles those NVIDIA libraries and
currently produces an approximately 377 MB wheel, above PyPI's default 100 MB
per-file limit. Strict manylinux CUDA wheels need either a PyPI file-size limit
increase or a packaging change that depends on external NVIDIA CUDA wheels and
sets a matching runtime library path.

## Subsequent versions

1. Update the version:

   ```sh
   make version VERSION=0.1.1
   ```

2. Run the tests you need for the change, commit the version bump, and merge it
   to `main`.
3. Run the GitHub `Release` workflow for that version, or publish locally with
   `make release` from a clean checkout.
