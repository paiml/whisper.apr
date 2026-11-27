#!/bin/sh
# readme-shape-falsifier — prove `pv-readme --check` can FAIL (#133, PMAT-136).
set -u

command -v pv-readme >/dev/null 2>&1 || {
    echo "NO-GO readme-shape-falsifier: pv-readme is not installed" >&2
    exit 2
}
[ -f README.md ] && [ -f contracts/readme-root.yaml ] || {
    echo "NO-GO readme-shape-falsifier: run from the repository root" >&2
    exit 2
}

work=$(mktemp -d) || exit 2
[ -n "$work" ] && [ -d "$work" ] && [ "$work" != "/" ] || exit 2
trap '[ -n "${work:?}" ] && [ "${work:?}" != "/" ] && rm -rf "${work:?}"' EXIT INT TERM
failed=0
cases=0

expect() {
    want=$1; needle=$2; label=$3; file=$4
    cases=$((cases + 1))
    out=$(pv-readme --check --readme "$file" --contracts contracts 2>&1)
    got=$?
    if [ "$got" -ne "$want" ]; then
        echo "FAIL  $label: exit $got, want $want"
        echo "$out" | head -n 5
        failed=$((failed + 1))
        return
    fi
    if [ -n "$needle" ]; then
        case $out in
            *"$needle"*) ;;
            *)
                echo "FAIL  $label: exit $got but the message does not name '$needle'"
                echo "$out" | head -n 5
                failed=$((failed + 1))
                return
                ;;
        esac
    fi
    echo "ok    $label (exit $got)"
}

expect 0 "" "the committed README.md conforms" README.md

cp README.md "$work/unrun.md"
printf '\n```bash\nwhisper-apr-readme-falsifier-never-run --planted\n```\n' >>"$work/unrun.md"
expect 1 "readme:verifiedCommand unresolved: whisper-apr-readme-falsifier-never-run --planted" \
    "an unrun bash claim is refused by name" "$work/unrun.md"

cp README.md "$work/text.md"
printf '\n```text\nwhisper-apr-readme-falsifier-never-run --planted\n```\n' >>"$work/text.md"
expect 0 "" "the same command in a text fence passes" "$work/text.md"

grep -v '^kind:' README.md >"$work/nokind.md"
expect 1 "" "a README without kind: is refused" "$work/nokind.md"

sed 's|^entrypoints: .*$|entrypoints: [readme-falsifier-absent-dir]|' README.md >"$work/noentry.md"
cmp -s README.md "$work/noentry.md" && {
    echo "NO-GO readme-shape-falsifier: the entrypoint plant changed nothing" >&2
    exit 2
}
expect 1 "" "an entrypoint that does not exist is refused" "$work/noentry.md"

sed 's|^contract_count: .*$|contract_count: 99999|' README.md >"$work/count.md"
cmp -s README.md "$work/count.md" && {
    echo "NO-GO readme-shape-falsifier: the count plant changed nothing" >&2
    exit 2
}
expect 1 "" "a contract count that is not the census is refused" "$work/count.md"

echo "readme-shape-falsifier: $cases case(s) run, $failed failed"
[ "$cases" -eq 6 ] || {
    echo "NO-GO readme-shape-falsifier: ran $cases case(s), want 6" >&2
    exit 2
}
[ "$failed" -eq 0 ]
