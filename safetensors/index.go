package safetensors

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/giraffesyo/ingot/tensor"
)

// Set is a checkpoint split over one or more files: either a single
// .safetensors file or shards listed by a *.safetensors.index.json.
type Set struct {
	files  []*File
	byName map[string]*File
}

// OpenDir opens the checkpoint in dir: the shards named by the directory's
// *.safetensors.index.json if there is one, else every *.safetensors file.
func OpenDir(dir string) (*Set, error) {
	idx, _ := filepath.Glob(filepath.Join(dir, "*.safetensors.index.json"))
	if len(idx) > 1 {
		return nil, fmt.Errorf("safetensors: %s: %d index files", dir, len(idx))
	}
	var paths []string
	if len(idx) == 1 {
		raw, err := os.ReadFile(idx[0])
		if err != nil {
			return nil, fmt.Errorf("safetensors: %w", err)
		}
		var ix struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := json.Unmarshal(raw, &ix); err != nil {
			return nil, fmt.Errorf("safetensors: %s: %w", idx[0], err)
		}
		seen := map[string]bool{}
		for _, f := range ix.WeightMap {
			if !seen[f] {
				seen[f] = true
				paths = append(paths, filepath.Join(dir, filepath.Clean(f)))
			}
		}
		sort.Strings(paths)
	} else {
		var err error
		if paths, err = filepath.Glob(filepath.Join(dir, "*.safetensors")); err != nil {
			return nil, err
		}
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("safetensors: no .safetensors files in %s", dir)
	}
	s := &Set{byName: map[string]*File{}}
	for _, p := range paths {
		f, err := Open(p)
		if err != nil {
			s.Close()
			return nil, err
		}
		s.files = append(s.files, f)
		for n := range f.tensors {
			if prev, dup := s.byName[n]; dup {
				s.Close()
				return nil, fmt.Errorf("safetensors: tensor %q in both %s and %s", n, prev.path, p)
			}
			s.byName[n] = f
		}
	}
	return s, nil
}

// Close closes every shard.
func (s *Set) Close() error {
	var errs []error
	for _, f := range s.files {
		errs = append(errs, f.Close())
	}
	return errors.Join(errs...)
}

// Names lists every tensor across shards, sorted.
func (s *Set) Names() []string {
	out := make([]string, 0, len(s.byName))
	for n := range s.byName {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// Info returns the header entry for name.
func (s *Set) Info(name string) (Info, bool) {
	f, ok := s.byName[name]
	if !ok {
		return Info{}, false
	}
	return f.Info(name)
}

// Tensor returns name in its stored dtype (see File.Tensor).
func (s *Set) Tensor(name string) (*tensor.Tensor, error) {
	f, ok := s.byName[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: no tensor %q in checkpoint", name)
	}
	return f.Tensor(name)
}

// F32 returns name as float32 (see File.F32).
func (s *Set) F32(name string) (*tensor.Tensor, error) {
	f, ok := s.byName[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: no tensor %q in checkpoint", name)
	}
	return f.F32(name)
}
