#!/bin/bash

echo "Building WASM version..."
./build_macroquad_wasm.sh

echo "Creating docs directory..."
mkdir -p docs

echo "Copying web files to docs..."
cp index.html docs/
cp pickover-attractor.wasm docs/

echo "Deployment files ready in docs/ directory!"
echo "To deploy:"
echo "1. Commit and push the docs/ folder"
echo "2. Enable GitHub Pages in repository settings"
echo "3. Set source to 'Deploy from a branch' → main → /docs" 