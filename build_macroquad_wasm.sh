#!/bin/bash

# Build script for pure Macroquad WASM version
set -e

echo "🚀 Building Pickover Attractor with pure Macroquad WASM..."

# Check if basic-http-server is installed (for local testing)
if ! command -v basic-http-server &> /dev/null; then
    echo "📦 Installing basic-http-server for local testing..."
    cargo install basic-http-server
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf pickover-attractor.wasm

# Build for WASM using cargo (macroquad's recommended approach)
echo "🔨 Building for WASM..."
cargo build --target wasm32-unknown-unknown --release

# Copy WASM file to root directory
echo "📁 Copying WASM file..."
cp target/wasm32-unknown-unknown/release/pickover-attractor.wasm .

echo "✅ Build complete!"
echo ""
echo "🌐 To run locally:"
echo "   basic-http-server ."
echo "   Then open http://localhost:4000"
echo ""
echo "📦 Files ready for deployment:"
echo "   - pickover-attractor.wasm"
echo "   - index.html" 