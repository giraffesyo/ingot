#!/bin/sh
# Checks commit subjects against Conventional Commits 1.0:
#
#   type(scope)!: description        scope and ! are optional
#
#   tools/commitlint.sh -f FILE      a commit message file (commit-msg hook)
#   tools/commitlint.sh -s SUBJECT   one subject line (a PR title)
#   tools/commitlint.sh RANGE        every non-merge commit in a rev range
set -eu

types='feat|fix|perf|docs|test|bench|refactor|build|ci|chore|style|revert'
pattern="^($types)(\([a-z0-9][a-z0-9_.,/-]*\))?!?: [^ .](.*[^ .])?$"

check() {
	case "$1" in
	"Merge "* | "Revert \""* | "fixup! "* | "squash! "* | "amend! "*) return 0 ;;
	esac
	if ! printf '%s\n' "$1" | grep -Eq "$pattern"; then
		echo "not a conventional commit: $1" >&2
		return 1
	fi
	if [ "${#1}" -gt 100 ]; then
		echo "subject longer than 100 characters: $1" >&2
		return 1
	fi
}

usage() {
	cat >&2 <<MSG

Expected: type(scope): description   e.g.  perf(gemm): 8x12 AVX-512 micro-kernel
Types:    $(echo "$types" | sed 's/|/, /g')
Append ! before the colon for a breaking change. No trailing period.
MSG
	exit 1
}

case "${1:-}" in
-f) check "$(sed -n '/^#/d;/./{p;q;}' "$2")" || usage ;;
-s) check "$2" || usage ;;
"") echo "usage: $0 -f FILE | -s SUBJECT | RANGE" >&2 && exit 2 ;;
*)
	bad=0
	git log --no-merges --format=%s "$1" >"${TMPDIR:-/tmp}/commitlint.$$"
	while IFS= read -r subject; do
		check "$subject" || bad=1
	done <"${TMPDIR:-/tmp}/commitlint.$$"
	rm -f "${TMPDIR:-/tmp}/commitlint.$$"
	[ "$bad" -eq 0 ] || usage
	;;
esac
