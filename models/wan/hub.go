package wan

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Repo is the Hugging Face repository this package reads: the Diffusers
// layout of Wan 2.2 TI2V-5B (Apache-2.0; umT5-XXL is Apache-2.0 too).
const Repo = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

// HubDir is the Hugging Face cache directory: $HF_HUB_CACHE, $HF_HOME/hub,
// or ~/.cache/huggingface/hub.
func HubDir() (string, error) {
	if hub := os.Getenv("HF_HUB_CACHE"); hub != "" {
		return hub, nil
	}
	if home := os.Getenv("HF_HOME"); home != "" {
		return filepath.Join(home, "hub"), nil
	}
	u, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("wan: %w", err)
	}
	return filepath.Join(u, ".cache", "huggingface", "hub"), nil
}

// FindSnapshot locates repo ("org/name") in the Hugging Face cache.
func FindSnapshot(repo string) (string, error) {
	hub, err := HubDir()
	if err != nil {
		return "", err
	}
	snaps, _ := filepath.Glob(filepath.Join(hub, "models--"+strings.ReplaceAll(repo, "/", "--"), "snapshots", "*"))
	for i := len(snaps) - 1; i >= 0; i-- {
		if entries, err := os.ReadDir(snaps[i]); err == nil && len(entries) > 0 {
			return snaps[i], nil
		}
	}
	return "", fmt.Errorf("wan: %s not found under %s (hf download %s, or pass its directory)", repo, hub, repo)
}
