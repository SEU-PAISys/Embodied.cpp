#!/usr/bin/env bash
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${ROOT}/third_party"

LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_REF="${LLAMA_REF:-b9016}"
DEFAULT_LLAMA_PATCHES=(
    "${ROOT}/patches/llama.cpp-pi05.patch"
    "${ROOT}/patches/llama.cpp-groot-n1.patch"
    "${ROOT}/patches/llama.cpp-cuda-parity.patch"
)

if [[ -n "${LLAMA_PATCHES:-}" ]]; then
    IFS=':' read -r -a LLAMA_PATCH_FILES <<< "${LLAMA_PATCHES}"
elif [[ -n "${LLAMA_PATCH:-}" ]]; then
    # Backward-compatible single-patch override.
    LLAMA_PATCH_FILES=("${LLAMA_PATCH}")
else
    LLAMA_PATCH_FILES=("${DEFAULT_LLAMA_PATCHES[@]}")
fi

mkdir -p "${THIRD_PARTY}"

checkout_ref() {
    local dir="$1"
    local repo="$2"
    local ref="$3"

    if git -C "${dir}" rev-parse --verify --quiet "${ref}^{commit}" >/dev/null; then
        git -C "${dir}" checkout --detach "${ref}"
        return
    fi

    if git -C "${dir}" fetch --depth 1 origin "${ref}" 2>/dev/null; then
        git -C "${dir}" checkout --detach FETCH_HEAD
        return
    fi

    echo "[third_party] ${ref} is not directly fetchable; fetching more history from ${repo}"
    git -C "${dir}" fetch --tags --depth 200 origin
    git -C "${dir}" checkout --detach "${ref}"
}

has_local_changes() {
    local dir="$1"
    [[ -n "$(git -C "${dir}" status --porcelain)" ]]
}

clone_or_update() {
    local name="$1"
    local repo="$2"
    local ref="$3"
    local dir="${THIRD_PARTY}/${name}"

    if [[ -d "${dir}/.git" ]]; then
        git -C "${dir}" remote set-url origin "${repo}"
        if has_local_changes "${dir}"; then
            echo "[third_party] ${name} has local changes; keeping existing checkout"
            return
        fi
        echo "[third_party] updating ${name}"
        checkout_ref "${dir}" "${repo}" "${ref}"
    elif [[ -e "${dir}" ]]; then
        echo "[third_party] ${dir} exists but is not a git checkout; leaving it unchanged"
    else
        echo "[third_party] cloning ${name} from ${repo} (${ref})"
        git clone --depth 1 "${repo}" "${dir}"
        checkout_ref "${dir}" "${repo}" "${ref}"
    fi
}

clone_or_update "llama.cpp" "${LLAMA_REPO}" "${LLAMA_REF}"

for patch in "${LLAMA_PATCH_FILES[@]}"; do
    patch_name="$(basename "${patch}")"
    if [[ ! -f "${patch}" ]]; then
        echo "[third_party] ERROR: llama.cpp patch not found: ${patch}" >&2
        exit 1
    fi
    if git -C "${THIRD_PARTY}/llama.cpp" apply --reverse --check "${patch}" 2>/dev/null; then
        echo "[third_party] ${patch_name} already applied"
    elif git -C "${THIRD_PARTY}/llama.cpp" apply --check "${patch}" 2>/dev/null; then
        git -C "${THIRD_PARTY}/llama.cpp" apply "${patch}"
        echo "[third_party] applied ${patch_name}"
    else
        echo "[third_party] ERROR: ${patch_name} does not apply to llama.cpp ${LLAMA_REF}" >&2
        exit 1
    fi
done

echo "[third_party] ready"
