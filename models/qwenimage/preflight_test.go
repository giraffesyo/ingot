//go:build darwin && arm64

package qwenimage

import (
	"image"
	"path/filepath"
	"strings"
	"testing"

	"github.com/giraffesyo/ingot/kernels/metal"
	"github.com/giraffesyo/ingot/tokenizer"
)

// TestPreflight: the stage estimates at 1024² and 2048², and a size that
// cannot fit fails before any stage runs.
func TestPreflight(t *testing.T) {
	if !metal.Available() {
		t.Skip("no Metal device")
	}
	dir := snapshotDir(t)
	tok, err := tokenizer.Load(filepath.Join(dir, "processor", "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode(promptText("a red fox in the snow", nil))
	if err != nil {
		t.Fatal(err)
	}
	sys, err := tok.Encode(sysMessage)
	if err != nil {
		t.Fatal(err)
	}
	for _, side := range []int{1024, 2048} {
		t.Logf("%d²:", side)
		if err := preflight(dir, ids, len(sys), nil, side/16, side/16, Options{Fast: true}, t.Logf); err != nil {
			t.Errorf("%d²: %v", side, err)
		}
	}
	// An edit at 1760×2368 (a portrait condition image at -size 2048): the
	// condition's ~16k latent tokens join the prefix.
	cw, ch := 1760, 2368
	conds := []condition{{img: image.NewNRGBA(image.Rect(0, 0, cw, ch)), gh: ch / visPatch, gw: cw / visPatch}}
	eids, err := tok.Encode(promptText("give the fox a red scarf", conds))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("edit %dx%d:", cw, ch)
	if err := preflight(dir, eids, len(sys), conds, ch/16, cw/16, Options{Fast: true}, t.Logf); err != nil {
		t.Errorf("edit %dx%d: %v", cw, ch, err)
	}
	err = preflight(dir, ids, len(sys), nil, 8192/16, 8192/16, Options{Fast: true}, t.Logf)
	if err == nil || !strings.Contains(err.Error(), "GPU memory") {
		t.Fatalf("8192²: preflight error %v, want an out-of-memory refusal", err)
	}
	t.Log(err)
}
