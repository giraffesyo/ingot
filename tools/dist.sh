#!/bin/sh
# Builds the release archives into dist/: every command in the module, static
# and cgo-free, one archive per target plus checksums.txt.
#
#   tools/dist.sh VERSION [os/arch ...]
set -eu

version=${1:?usage: tools/dist.sh VERSION [os/arch ...]}
shift
[ $# -gt 0 ] || set -- linux/amd64 linux/arm64 darwin/amd64 darwin/arm64 windows/amd64 windows/arm64

export CGO_ENABLED=0 GOWORK=off GOFLAGS=-trimpath
cmds=$(go list -f '{{if eq .Name "main"}}{{.ImportPath}}{{end}}' ./cmd/...)

rm -rf dist
mkdir dist
for target; do
	os=${target%/*} arch=${target#*/}
	name=ingot_${version#v}_${os}_${arch}
	echo "== $name"
	mkdir "dist/$name"
	# shellcheck disable=SC2086 # one argument per command
	GOOS=$os GOARCH=$arch go build -ldflags='-s -w -buildid=' -o "dist/$name/" $cmds
	cp LICENSE NOTICE README.md "dist/$name/"
	if [ "$os" = windows ]; then
		(cd dist && zip -qrX "$name.zip" "$name")
	else
		tar -C dist -czf "dist/$name.tar.gz" "$name"
	fi
	rm -r "dist/$name"
done

cd dist
if command -v sha256sum >/dev/null; then
	sha256sum -- * >checksums.txt
else
	shasum -a 256 -- * >checksums.txt
fi
cat checksums.txt
