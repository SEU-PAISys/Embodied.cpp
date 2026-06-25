#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${ROOT}/third_party"

LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_REF="${LLAMA_REF:-b9016}"
LLAMA_PATCH="${LLAMA_PATCH:-${ROOT}/patches/llama.cpp-vla.patch}"

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

if [[ -f "${LLAMA_PATCH}" ]]; then
    if git -C "${THIRD_PARTY}/llama.cpp" apply --reverse --check "${LLAMA_PATCH}" 2>/dev/null; then
        echo "[third_party] llama.cpp patch already applied"
    elif git -C "${THIRD_PARTY}/llama.cpp" apply --check "${LLAMA_PATCH}" 2>/dev/null; then
        git -C "${THIRD_PARTY}/llama.cpp" apply "${LLAMA_PATCH}"
        echo "[third_party] applied llama.cpp patch"
    else
        echo "[third_party] ERROR: ${LLAMA_PATCH} does not apply to llama.cpp ${LLAMA_REF}" >&2
        exit 1
    fi
else
    echo "[third_party] warning: llama.cpp patch not found at ${LLAMA_PATCH}; skipping"
fi

echo "[third_party] ready"
