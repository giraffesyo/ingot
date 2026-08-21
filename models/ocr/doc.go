// Package ocr implements the end-to-end OCR pipeline on the pure-Go runtime:
// image preprocessing, DBNet-style text detection with differentiable-
// binarization post-processing, and (planned) CRNN/SVTR recognition.
//
// Models are standard PP-OCR ONNX files loaded via the graph package. Nothing
// here depends on a specific model beyond input/output tensor conventions.
package ocr
