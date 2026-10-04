package main

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

func resolveFolder(path, fallback string) (string, error) {
	if path == "" {
		path = fallback
	}
	if strings.IndexByte(path, 0) >= 0 {
		return "", fmt.Errorf("folder paths cannot contain a null character")
	}
	if path == "~" || strings.HasPrefix(path, "~/") || strings.HasPrefix(path, `~\`) {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		path = filepath.Join(home, strings.TrimLeft(path[1:], `/\`))
	}
	return filepath.Abs(path)
}

type folderListing struct {
	Path    string   `json:"path"`
	Parent  string   `json:"parent"`
	Folders []string `json:"folders"`
}

// List directory names only. The browser never receives arbitrary file contents.
func (a *app) folders(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("X-Ingot-UI") != "1" {
		apiError(w, http.StatusForbidden, "Missing UI request header.")
		return
	}
	path, err := resolveFolder(r.URL.Query().Get("path"), a.output)
	if err != nil {
		apiError(w, http.StatusBadRequest, "Choose a valid folder: "+err.Error())
		return
	}
	entries, err := os.ReadDir(path)
	if err != nil {
		apiError(w, http.StatusBadRequest, "Could not open this folder. Check the path and permissions, or enter a new output path in the form.")
		return
	}
	listing := folderListing{Path: path, Parent: filepath.Dir(path), Folders: []string{}}
	for _, entry := range entries {
		isDir := entry.IsDir()
		if entry.Type()&os.ModeSymlink != 0 {
			info, err := os.Stat(filepath.Join(path, entry.Name()))
			isDir = err == nil && info.IsDir()
		}
		if isDir {
			listing.Folders = append(listing.Folders, entry.Name())
		}
	}
	writeJSON(w, http.StatusOK, listing)
}
