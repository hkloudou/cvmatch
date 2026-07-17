.PHONY: all test regolden

all: test

test:
	go test ./...

# Deliberate golden re-record (the ONLY way constants change): native
# tolerance parity runs FIRST — an arithmetic change that violates the
# OpenCV tolerance contract can never be recorded — then the table is
# rewritten, then the full cross-arch self-identity matrix reproves the
# new constants. Commit with a "Goldens:" trailer explaining the change.
regolden:
	@test -n "$(REASON)" || { echo 'REASON is required: make regolden REASON="..."'; exit 1; }
	cd bench && ./testdata/fetch.sh
	cd bench && go run ./cmd/dumpscenes -out cpp/scenes && cpp/build.sh
	cd bench && cpp/native_bench cpp/scenes 1 dump >/dev/null
	cd bench && go test -run 'TestFullMapParityWithNativeCpp|TestNativeValues' -count=1 .
	go test -run TestGoldenOutputs -count=1 . -args -cvmatch.record -cvmatch.reason "$(REASON)"
	go test -count=1 . && go test -tags purego -count=1 .
	go test -race -count=1 . && go test -tags purego -race -count=1 .
	GOOS=linux GOARCH=arm64 go test -c -o /tmp/cvmatch-regolden.arm64 . \
		&& qemu-aarch64-static /tmp/cvmatch-regolden.arm64
	GOOS=linux GOARCH=arm64 go test -tags purego -c -o /tmp/cvmatch-regolden-purego.arm64 . \
		&& qemu-aarch64-static /tmp/cvmatch-regolden-purego.arm64
	@echo 'Recorded and cross-arch proven. Commit with a "Goldens:" trailer.'
