// Command ingot runs models on the ingot runtime: any ONNX file, OCR, and
// the generative pipelines (image, video, speech, audio, 3D), all pure Go.
//
//	ingot run --model m.onnx --random
//	ingot ocr --in page.png --format md
//	ingot qwenimage --prompt "a red fox in the snow" --out fox.png
//
// Shell completions: ingot completion bash|zsh|fish|powershell.
package main

import (
	"fmt"
	"os"
	"runtime/debug"

	"github.com/spf13/cobra"

	"github.com/giraffesyo/ingot/cmd/ingot/internal/ocrcmd"
	"github.com/giraffesyo/ingot/cmd/ingot/internal/prosodycmd"
	"github.com/giraffesyo/ingot/cmd/ingot/internal/qwen3ttscmd"
	"github.com/giraffesyo/ingot/cmd/ingot/internal/qwenimagecmd"
	"github.com/giraffesyo/ingot/cmd/ingot/internal/runcmd"
	"github.com/giraffesyo/ingot/cmd/ingot/internal/stableaudiocmd"
	"github.com/giraffesyo/ingot/cmd/ingot/internal/trellis2cmd"
	"github.com/giraffesyo/ingot/cmd/ingot/internal/wancmd"
)

// version is set at release time with -ldflags '-X main.version=vX.Y.Z'.
var version string

func main() {
	if err := newRoot().Execute(); err != nil {
		fmt.Fprintln(os.Stderr, "ingot:", err)
		os.Exit(1)
	}
}

func newRoot() *cobra.Command {
	root := &cobra.Command{
		Use:           "ingot",
		Short:         "Pure-Go ONNX and model inference runtime",
		Long:          "ingot runs ONNX models and Go-defined pipelines over Hugging Face checkpoints\non the CPU (and the Apple GPU where available), with no cgo and no external runtime.",
		Version:       buildVersion(),
		SilenceUsage:  true,
		SilenceErrors: true,
	}
	root.AddCommand(
		runcmd.Command(),
		ocrcmd.Command(),
		qwenimagecmd.Command(),
		wancmd.Command(),
		trellis2cmd.Command(),
		qwen3ttscmd.Command(),
		stableaudiocmd.Command(),
		prosodycmd.Command(),
		&cobra.Command{
			Use:   "version",
			Short: "Print the version",
			Args:  cobra.NoArgs,
			Run: func(cmd *cobra.Command, _ []string) {
				cmd.Println("ingot", buildVersion())
			},
		},
	)
	return root
}

// buildVersion is the release version, or the module version for go install.
func buildVersion() string {
	if version != "" {
		return version
	}
	if bi, ok := debug.ReadBuildInfo(); ok && bi.Main.Version != "" {
		return bi.Main.Version
	}
	return "(devel)"
}
