# Pickover Attractor

A sophisticated Rust application using macroquad to visualize Pickover attractors with multiple color channels, real-time parameter generation, and interactive controls.

## Features

- **Multiple Color Modes**: RGB (independent channels), Monochrome (grayscale), and Correlated (related channels)
- **Real-time Parameter Generation**: Automatically generates interesting attractor parameters
- **Interactive Controls**: UI for switching color modes and adjusting correlated deviation
- **Performance Monitoring**: Real-time FPS and performance metrics
- **Window Resizing**: Handles dynamic window resizing
- **Automatic Reset**: Resets when attractors become saturated or stagnant

## Controls

- **Space**: Generate new attractor parameters
- **Q**: Quit the application
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

### Prerequisites
- Rust and Cargo installed on your system

### Build
```bash
cargo build
```

### Run
```bash
cargo run
```

### Release Build
```bash
cargo build --release
```

## Performance

The application is optimized for real-time visualization:
- Efficient pixel buffer management
- Change tracking to minimize texture updates
- Automatic performance monitoring and logging

## Project Structure

```
pickover/
├── Cargo.toml
├── src/
│   └── main.rs
└── README.md
```

## Dependencies

- `macroquad = "0.4"` - Cross-platform game framework for Rust

## License

This project is derived from the macroquad examples and follows the same licensing terms. 