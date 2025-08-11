# Pickover Attractor

A Rust application using [macroquad](https://github.com/not-fl3/macroquad) to visualize Pickover attractors with multiple color channels, real-time parameter generation, interactive controls, and beautiful symmetry patterns. **Runs on both desktop and web browsers using WebAssembly!**

🌐 **Live Demo**: [https://dmaynard.github.io/pickover-attractor](https://dmaynard.github.io/pickover-attractor)

## Features

- **Multiple Color Modes**: RGB (independent channels), Monochrome (grayscale), and Correlated (related channels)
- **Day/Night Mode**: Toggle between light and dark themes with proper visual clearing
- **Symmetry Patterns**: 4-fold, 6-fold, and 8-fold radial symmetry for stunning geometric patterns
- **Real-time Parameter Generation**: Automatically generates interesting attractor parameters
- **Interactive Controls**: UI for switching color modes, symmetry, and adjusting correlated deviation
- **Performance Monitoring**: Real-time FPS and performance metrics
- **Window Resizing**: Handles dynamic window resizing
- **Automatic Reset**: Resets when attractors become saturated or stagnant
- **Cross-Platform**: Runs natively on desktop and in web browsers via WebAssembly

## Controls

### Keyboard Controls
- **Space**: Generate new attractor parameters
- **I**: Toggle day/night mode (invert colors)
- **M**: Cycle through color modes (RGB → Monochrome → Correlated)
- **S**: Cycle through symmetry modes (None → 4-fold → 6-fold → 8-fold)
- **R**: Toggle red channel on/off (RGB mode only)
- **G**: Toggle green channel on/off (RGB mode only)
- **B**: Toggle blue channel on/off (RGB mode only)
- **Slash (/)**: Toggle help display
- **Q**: Quit the application (desktop only)

### Mouse Controls
- **Left Click**: Click UI buttons for color mode, symmetry, day/night toggle, and next attractor
- **Drag**: Adjust the correlated deviation slider (in correlated mode)

## Color Modes

### RGB Mode
- Each color channel (Red, Green, Blue) operates independently
- Each channel has its own attractor parameters
- Individual channels can be toggled on/off with R, G, B keys
- Creates colorful, dynamic visualizations

### Monochrome Mode
- Only the red channel is active
- Creates grayscale visualizations
- Useful for studying the mathematical structure

### Correlated Mode
- All channels share base parameters with small deviations
- Adjustable deviation percentage (0-5%)
- Creates harmonious, related color patterns

## Symmetry Modes

### No Symmetry
- Shows the natural, organic form of the attractor
- Best for studying the base mathematical behavior

### 4-Fold Radial Symmetry
- Creates 90-degree rotational patterns
- Produces geometric, mandala-like designs

### 6-Fold Radial Symmetry
- Creates 60-degree rotational patterns
- Generates snowflake-like crystalline structures

### 8-Fold Radial Symmetry
- Creates 45-degree rotational patterns
- Produces complex, kaleidoscopic patterns

## Day/Night Mode

- **Day Mode**: Black background with bright attractor colors
- **Night Mode**: White background with dark attractor colors
- Seamlessly switches between modes without visual artifacts
- All new attractors inherit the current mode setting

## Technical Details

The application implements the Pickover attractor equations:
- `x' = sin(b*y) - c*sin(b*x)`
- `y' = sin(a*x) + d*cos(a*y)`

Where a, b, c, and d are parameters that determine the attractor's behavior.

The symmetry calculations use high-precision floating-point arithmetic to ensure smooth, artifact-free patterns.

## Building and Running

### Desktop Version

#### Prerequisites
- Rust and Cargo installed on your system

#### Build
```bash
cargo build
```

#### Run
```bash
cargo run
```

#### Release Build
```bash
cargo build --release
cargo run --release
```

### Web Version (WebAssembly)

#### Prerequisites
- Rust and Cargo installed on your system
- A modern web browser (Chrome, Firefox, Safari, or Edge)

#### Build and Test Locally

**Step 1: Build the WASM version**
```bash
# Make the build script executable (first time only)
chmod +x build_macroquad_wasm.sh

# Build the WebAssembly version
./build_macroquad_wasm.sh
```

**Step 2: Start a local web server**

*Option A: Using basic-http-server (recommended)*
```bash
# The build script should install this automatically
basic-http-server .
# Then open http://localhost:4000 in your browser
```

*Option B: Using Python (if you have Python 3 installed)*
```bash
python3 -m http.server 4000
# Then open http://localhost:4000 in your browser
```

*Option C: Using Node.js (if you have Node.js installed)*
```bash
npx http-server . -p 4000
# Then open http://localhost:4000 in your browser
```

**Step 3: Test in your browser**
- Open your web browser
- Navigate to `http://localhost:4000`
- You should see the Pickover Attractor running in your browser!
- Try the keyboard controls: Space (new attractor), I (day/night), M (color modes), S (symmetry)

#### Troubleshooting
- **Port 4000 already in use**: Try a different port (e.g., `basic-http-server . --port 4001`)
- **Build fails**: Make sure you have the latest Rust toolchain (`rustup update`)
- **WASM doesn't load**: Check browser console for errors and ensure you're using a modern browser
- **Performance issues**: Close other browser tabs and ensure hardware acceleration is enabled

## Performance

The application is optimized for real-time visualization:
- Efficient pixel buffer management
- Change tracking to minimize texture updates
- Automatic performance monitoring and logging
- **WebAssembly provides near-native performance in browsers**

## Project Structure

```
pickover/
├── Cargo.toml
├── src/
│   └── main.rs          # Main application code (works for both desktop and web)
├── index.html           # Web interface
├── build_macroquad_wasm.sh  # WASM build script
└── README.md
```

## Dependencies

- `macroquad = "0.4"` - Cross-platform game framework for Rust (supports WASM)

## Browser Compatibility

- Chrome 57+
- Firefox 52+
- Safari 11+
- Edge 79+

## Deployment

### Desktop
- Build with `cargo build --release` and distribute the binary

### Web
- Run `./build_macroquad_wasm.sh` to generate the WASM file
- Deploy `index.html` and `pickover-attractor.wasm` to any static hosting service (GitHub Pages, Netlify, Vercel, etc.)

## Credits

This project was developed with assistance from Claude Sonnet 4, an AI coding assistant from Anthropic, used through the Cursor IDE. The AI helped with:

- Initial project structure and Rust/macroquad setup
- Implementation of the Pickover attractor equations
- Multi-channel color system (RGB, Monochrome, Correlated modes)
- WebAssembly compilation and deployment
- Performance optimizations and real-time visualization
- UI/UX improvements and interactive controls
- Cross-platform compatibility (desktop and web)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2024 David Maynard

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 