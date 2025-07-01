# Pickover Attractor

A sophisticated Rust application using macroquad to visualize Pickover attractors with multiple color channels, real-time parameter generation, and interactive controls. **Runs on both desktop and web browsers using WebAssembly!**

## Features

- **Multiple Color Modes**: RGB (independent channels), Monochrome (grayscale), and Correlated (related channels)
- **Real-time Parameter Generation**: Automatically generates interesting attractor parameters
- **Interactive Controls**: UI for switching color modes and adjusting correlated deviation
- **Performance Monitoring**: Real-time FPS and performance metrics
- **Window Resizing**: Handles dynamic window resizing
- **Automatic Reset**: Resets when attractors become saturated or stagnant
- **Cross-Platform**: Runs natively on desktop and in web browsers via WebAssembly

## Controls

- **Space**: Generate new attractor parameters
- **Q**: Quit the application (desktop only)
- **Slash (/)**: Toggle help display
- **Mouse**: Click and drag the correlated deviation slider (in correlated mode)

## Color Modes

### RGB Mode
- Each color channel (Red, Green, Blue) operates independently
- Each channel has its own attractor parameters
- Creates colorful, dynamic visualizations

### Monochrome Mode
- Only the red channel is active
- Creates grayscale visualizations
- Useful for studying the mathematical structure

### Correlated Mode
- All channels share base parameters with small deviations
- Adjustable deviation percentage (0-5%)
- Creates harmonious, related color patterns

## Technical Details

The application implements the Pickover attractor equations:
- `x' = sin(b*y) - c*sin(b*x)`
- `y' = sin(a*x) + d*cos(a*y)`

Where a, b, c, and d are parameters that determine the attractor's behavior.

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
- Rust and Cargo
- `basic-http-server` (will be installed automatically by the build script)

#### Build and Run
```bash
# Build the WASM version
./build_macroquad_wasm.sh

# Run locally
basic-http-server .
# Then open http://localhost:4000 in your browser
```

#### Alternative using Python:
```bash
# After building with ./build_macroquad_wasm.sh
python3 -m http.server 4000
# Then open http://localhost:4000 in your browser
```

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

## License

This project is derived from the macroquad examples and follows the same licensing terms. 