// Package onnx decodes ONNX model files (protobuf wire format) into plain Go
// structs, without a protobuf dependency. Only the subset of onnx.proto needed
// for inference is decoded; unknown fields are skipped.
//
// Field numbers follow onnx/onnx.proto3 (IR version 10).
package onnx
