// Command qwenimage-ui is an optional, offline browser UI with a stage-batched
// image queue. Run this separate module with go run .
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	listen := flag.String("listen", "127.0.0.1:7860", "loopback address for the local UI")
	model := flag.String("model", "", "local Qwen-Image-2.1 snapshot (default: CLI's Hugging Face cache)")
	output := flag.String("output", "outputs", "directory for generated PNGs and recoverable latents")
	flag.Parse()
	addr, err := loopbackAddress(*listen)
	if err != nil {
		return err
	}
	if *model == "" {
		*model, err = findSnapshot()
		if err != nil {
			return err
		}
	}
	outputDir, err := filepath.Abs(*output)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(outputDir, 0700); err != nil {
		return fmt.Errorf("create output directory: %w", err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	app := newApp(ctx, *model, outputDir)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	app.startWorker()
	server := &http.Server{
		Handler: app.handler(listener.Addr().String()), ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout: 2 * time.Minute, WriteTimeout: 30 * time.Second, IdleTimeout: time.Minute,
		MaxHeaderBytes: 16 << 10,
	}
	done := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_ = server.Shutdown(shutdown)
		case <-done:
		}
	}()
	fmt.Printf("QwenImage UI: http://%s\nOutputs: %s\nPress Ctrl+C to stop.\n", listener.Addr(), outputDir)
	err = server.Serve(listener)
	close(done)
	stop()
	app.wait.Wait()
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}

func findSnapshot() (string, error) {
	hub := os.Getenv("HF_HUB_CACHE")
	if hub == "" {
		if home := os.Getenv("HF_HOME"); home != "" {
			hub = filepath.Join(home, "hub")
		} else {
			home, err := os.UserHomeDir()
			if err != nil {
				return "", err
			}
			hub = filepath.Join(home, ".cache", "huggingface", "hub")
		}
	}
	snaps, _ := filepath.Glob(filepath.Join(hub, "models--Qwen--Qwen-Image-2.1", "snapshots", "*"))
	if len(snaps) == 0 {
		return "", fmt.Errorf("Qwen-Image-2.1 not found under %s; provide local weights with -model", hub)
	}
	return snaps[len(snaps)-1], nil
}

func loopbackAddress(addr string) (string, error) {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return "", fmt.Errorf("listen address: %w", err)
	}
	if host == "localhost" {
		host = "127.0.0.1"
	}
	ip := net.ParseIP(host)
	if ip == nil || !ip.IsLoopback() {
		return "", fmt.Errorf("listen address must be a loopback IP or localhost")
	}
	return net.JoinHostPort(host, port), nil
}
