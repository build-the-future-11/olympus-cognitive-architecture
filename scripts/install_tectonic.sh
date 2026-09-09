#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 INSTALL_DIRECTORY" >&2
  exit 2
fi

tectonic_version="0.17.0"
platform="$(uname -s)-$(uname -m)"
case "${platform}" in
  Darwin-arm64)
    target="aarch64-apple-darwin"
    expected_sha256="a3f1cac7c5678f01661a92212f58480ae3b0634115d880dbc59e2953ded45667"
    ;;
  Darwin-x86_64)
    target="x86_64-apple-darwin"
    expected_sha256="7c90ef5b6ddb1eb1937e4337add5237b79338e4b9676459fa91187d24d6cdf80"
    ;;
  Linux-x86_64)
    target="x86_64-unknown-linux-gnu"
    expected_sha256="1a715688baf591e650c8aeb160ae934e181685eecbb38b317de30b269ac5d606"
    ;;
  *)
    echo "unsupported Tectonic release platform: ${platform}" >&2
    exit 2
    ;;
esac

install_directory="$1"
temporary_directory="$(mktemp -d "${TMPDIR:-/tmp}/olympus-tectonic.XXXXXX")"
trap 'rm -rf "${temporary_directory}"' EXIT

archive_name="tectonic-${tectonic_version}-${target}.tar.gz"
archive_path="${temporary_directory}/${archive_name}"
download_url="https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40${tectonic_version}/${archive_name}"

curl --fail --location --proto '=https' --tlsv1.2 \
  --output "${archive_path}" "${download_url}"
printf '%s  %s\n' "${expected_sha256}" "${archive_path}" | shasum -a 256 --check
tar -xzf "${archive_path}" -C "${temporary_directory}"
test "$("${temporary_directory}/tectonic" --version)" = "Tectonic ${tectonic_version}"

install -d -m 0755 "${install_directory}"
install -m 0755 "${temporary_directory}/tectonic" "${install_directory}/tectonic"
