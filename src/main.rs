use macroquad::prelude::*;
use macroquad::ui::{hash, root_ui};

const SCALE: f64 = 120.0;  // Adjusted scale for better visibility
const MAX_ATTRACTOR_RUNTIME: f64 = 60.0;  // Maximum time in seconds before forcing a reset
const DISPLAY_DURATION: f64 = 10.0;  // Time to display the attractor in seconds
const FADE_DURATION: f64 = 4.0;  // Time to fade out the attractor in seconds
const SATURATION_THRESHOLD: f64 = 0.15;  // Reset when 15% of active pixels are maxed out
const DEFAULT_CORRELATED_DEVIATION: f64 = 0.002;  // Default percentage deviation for correlated mode parameters (0.002 = 0.2% deviation)
const UI_AREA_HEIGHT: f32 = 100.0;  // Height reserved for UI elements at the bottom (increased from 75.0)

#[derive(PartialEq, Clone)]
enum AttractorStatus {
    Running,    // Actively calculating new points
    Displaying, // Showing the current state without updates
    Fading,     // Gradually fading out the current state
}

#[derive(Clone, Copy, Debug)]
enum ColorChannel {
    Red,
    Green,
    Blue,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ColorState {
    RGB,        // Independent RGB channels
    Monochrome, // Single channel (grayscale)
    Correlated, // Channels are correlated/related to each other
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SymmetryType {
    None,    // No symmetry
    Radial4, // 4-fold radial symmetry
    Radial6, // 6-fold radial symmetry
    Radial8, // 8-fold radial symmetry
}

#[derive(Clone)]
struct PickoverSystem {
    x: f64,
    y: f64,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    scalex: f64,
    scaley: f64,
    minx: f64,
    miny: f64,
    pixels: Vec<u8>,
    changed_pixel_indices: Vec<usize>,  // Only store indices of changed pixels
    needs_full_refresh: bool,           // Flag for when entire image needs refresh
    symmetry_buffer: Vec<(i32, i32)>,  // Pre-allocated buffer for symmetry calculations
    width: usize,
    height: usize,
    nonzero_pixels: usize,  // Count of pixels that are non-zero
    maxed_pixels: usize,    // Count of pixels that have reached 255
    invert: bool,  // Added invert flag
    status: AttractorStatus,
    display_start_time: f64,    // When display phase started
    fade_start_time: f64,       // When fade phase started
    channel: ColorChannel,  // Which color channel to use
    active: bool,          // Whether this attractor is currently active
    start_time: f64,       // When this attractor configuration started running
    monochrome: bool,      // Whether to use monochrome mode
    color_state: ColorState, // Current color state
    symmetry: SymmetryType, // Symmetry transformation to apply
    consecutive_empty_steps: u32, // Counter for debugging empty pixel updates
}

// Pre-calculated symmetry transformation values for performance
struct SymmetryLookupTables {
    radial4: Vec<(f64, f64)>,  // (cos, sin) pairs for 4-fold symmetry
    radial6: Vec<(f64, f64)>,  // (cos, sin) pairs for 6-fold symmetry  
    radial8: Vec<(f64, f64)>,  // (cos, sin) pairs for 8-fold symmetry
}

impl SymmetryLookupTables {
    fn new() -> Self {
        let mut radial4 = Vec::new();
        let mut radial6 = Vec::new();
        let mut radial8 = Vec::new();
        
        // 4-fold symmetry: 90°, 180°, 270° rotations
        for i in 1..4 {
            let angle = i as f64 * std::f64::consts::PI / 2.0;
            radial4.push((angle.cos(), angle.sin()));
        }
        
        // 6-fold symmetry: 60°, 120°, 180°, 240°, 300° rotations
        for i in 1..6 {
            let angle = i as f64 * std::f64::consts::PI / 3.0;
            radial6.push((angle.cos(), angle.sin()));
        }
        
        // 8-fold symmetry: 45°, 90°, 135°, 180°, 225°, 270°, 315° rotations
        for i in 1..8 {
            let angle = i as f64 * std::f64::consts::PI / 4.0;
            radial8.push((angle.cos(), angle.sin()));
        }
        
        Self { radial4, radial6, radial8 }
    }
}

// Global lookup tables (initialized once)
static mut SYMMETRY_TABLES: Option<SymmetryLookupTables> = None;

fn get_symmetry_tables() -> &'static SymmetryLookupTables {
    unsafe {
        SYMMETRY_TABLES.as_ref().unwrap()
    }
}

impl PickoverSystem {
    // Helper function to apply deviation to a parameter
    fn apply_deviation(base_value: f64, deviation_percent: f64) -> f64 {
        if deviation_percent == 0.0 {
            base_value
        } else {
            let deviation = (rand::gen_range(-100.0, 100.0) as f64) / 100.0 * deviation_percent;
            base_value * (1.0 + deviation)
        }
    }

    // Extract attractor equation into its own function
    fn next_point(&self, x: f64, y: f64) -> (f64, f64) {
        let new_x = (self.b * y).sin() - self.c * (self.b * x).sin();
        let new_y = (self.a * x).sin() + self.d * (self.a * y).cos();
        (new_x, new_y)
    }

    // Get the effective drawing bounds (square for symmetry modes, full rectangle for none)
    fn get_drawing_bounds(&self) -> (f64, f64, f64, f64) {
        if self.symmetry == SymmetryType::None {
            // Use full window for no symmetry
            (0.0, 0.0, self.width as f64, self.height as f64)
        } else {
            // Use centered square for symmetry modes - keep as f64 for precision
            let square_size = self.width.min(self.height) as f64;
            let offset_x = (self.width as f64 - square_size) * 0.5;
            let offset_y = (self.height as f64 - square_size) * 0.5;
            (offset_x, offset_y, offset_x + square_size, offset_y + square_size)
        }
    }

    // Generate symmetric pixel coordinates for a given point (f64 version for maximum precision)
    fn get_symmetric_pixels_f64(&self, screen_x: f64, screen_y: f64) -> Vec<(i32, i32)> {
        let mut pixels = Vec::new();
        
        // Get the effective drawing bounds (now in f64 for precision)
        let (min_x, min_y, max_x, max_y) = self.get_drawing_bounds();
        
        // For symmetry modes, we need to transform from full-window coordinates to centered-square coordinates
        if self.symmetry != SymmetryType::None {
            // Transform the original point to the centered square (keep as f64 for precision)
            let transformed_x = screen_x + min_x;
            let transformed_y = screen_y + min_y;
            
            // Calculate center of the drawing area for transformations (keep as f64)
            let center_x = (min_x + max_x) * 0.5;
            let center_y = (min_y + max_y) * 0.5;
            
            // Translate to center-based coordinates (keep as f64 for precision)
            let rel_x = transformed_x - center_x;
            let rel_y = transformed_y - center_y;
            
            // Add the original transformed pixel
            pixels.push((transformed_x.round() as i32, transformed_y.round() as i32));
        
        match self.symmetry {
            SymmetryType::None => {
                // No additional pixels - coordinates are already correct
            }
            SymmetryType::Radial4 => {
                // 4-fold radial symmetry (90-degree rotations around center)
                let tables = get_symmetry_tables();
                for &(cos_angle, sin_angle) in &tables.radial4 {
                    let rotated_x = (rel_x * cos_angle - rel_y * sin_angle).round() as i32 + center_x.round() as i32;
                    let rotated_y = (rel_x * sin_angle + rel_y * cos_angle).round() as i32 + center_y.round() as i32;
                    pixels.push((rotated_x, rotated_y));
                }
            }
            SymmetryType::Radial6 => {
                // 6-fold radial symmetry (60-degree rotations around center)
                let tables = get_symmetry_tables();
                for &(cos_angle, sin_angle) in &tables.radial6 {
                    let rotated_x = (rel_x * cos_angle - rel_y * sin_angle).round() as i32 + center_x.round() as i32;
                    let rotated_y = (rel_x * sin_angle + rel_y * cos_angle).round() as i32 + center_y.round() as i32;
                    pixels.push((rotated_x, rotated_y));
                }
            }
            SymmetryType::Radial8 => {
                // 8-fold radial symmetry (45-degree rotations around center)
                let tables = get_symmetry_tables();
                for &(cos_angle, sin_angle) in &tables.radial8 {
                    let rotated_x = (rel_x * cos_angle - rel_y * sin_angle).round() as i32 + center_x.round() as i32;
                    let rotated_y = (rel_x * sin_angle + rel_y * cos_angle).round() as i32 + center_y.round() as i32;
                    pixels.push((rotated_x, rotated_y));
                }
            }
        }
        } else {
            // No symmetry: just add the original pixel
            pixels.push((screen_x.round() as i32, screen_y.round() as i32));
        }
        
        pixels
    }

    // Generate symmetric pixel coordinates for a given point (legacy i32 version)
    fn get_symmetric_pixels(&self, screen_x: i32, screen_y: i32) -> Vec<(i32, i32)> {
        self.get_symmetric_pixels_f64(screen_x as f64, screen_y as f64)
    }

    // Extract common image clearing code
    fn clear_image(&mut self, image_data: &mut [[u8; 4]], fill_value: u8) {
        image_data.iter_mut().for_each(|pixel| {
            pixel[0] = fill_value;
            pixel[1] = fill_value;
            pixel[2] = fill_value;
            pixel[3] = 255;
        });
        self.changed_pixel_indices.clear();
    }

    fn step(&mut self) {
        let (new_x, new_y) = self.next_point(self.x, self.y);
        self.x = new_x;
        self.y = new_y;

        // Calculate screen coordinates (keep as f64 for precision until final conversion)
        let screen_x_f64 = (self.x - self.minx) * self.scalex;
        let screen_y_f64 = (self.y - self.miny) * self.scaley;

        // Get drawing bounds for offset calculation (now in f64 for precision)
        let _bounds = self.get_drawing_bounds(); // Used by get_symmetric_pixels_f64

        // Get all symmetric pixel coordinates (pass f64 coordinates)
        let symmetric_pixels = self.get_symmetric_pixels_f64(screen_x_f64, screen_y_f64);
        let total_symmetric_pixels = symmetric_pixels.len();
        
        // Plot all symmetric pixels
        let mut pixels_plotted_this_step = 0;
        
        for (px, py) in symmetric_pixels {
            if px >= 0 && px < self.width as i32 && py >= 0 && py < self.height as i32 {
                let idx = py as usize * self.width + px as usize;
                if idx < self.pixels.len() {
                    let old_value = self.pixels[idx];
                    if old_value == 0 {
                        self.nonzero_pixels += 1;
                    }
                    
                    // Ensure we never wrap from 255 to 0 by using saturating_add
                    let new_value = if old_value == 255 {
                        255  // Already at max, stay there
                    } else if old_value == 254 {
                        self.maxed_pixels += 1;
                        255
                    } else {
                        old_value.saturating_add(1)  // This will never overflow
                    };

                    if new_value != old_value {
                        self.pixels[idx] = new_value;
                        self.changed_pixel_indices.push(idx);
                        pixels_plotted_this_step += 1;
                    }
                }
            }
        }
        
        // Debug: Only print when we're not plotting any pixels for multiple steps
        if pixels_plotted_this_step == 0 {
            self.consecutive_empty_steps += 1;
            if self.consecutive_empty_steps >= 1000 {  // Only warn after 1000 consecutive empty steps
                println!("  Warning: Attractor {} has plotted 0 pixels for {} consecutive steps", 
                         match self.channel { ColorChannel::Red => "Red", ColorChannel::Green => "Green", ColorChannel::Blue => "Blue" },
                         self.consecutive_empty_steps);
                println!("    Current coords: x={:.3}, y={:.3}, screen_x={:.1}, screen_y={:.1}", 
                         self.x, self.y, screen_x_f64, screen_y_f64);
                println!("    Scales: scalex={:.6}, scaley={:.6}", self.scalex, self.scaley);
                println!("    Symmetric pixels generated: {}", total_symmetric_pixels);
                
                // Check if the pixels are already maxed out
                if total_symmetric_pixels > 0 {
                    let first_px = (screen_x_f64.round() as i32, screen_y_f64.round() as i32);
                    if first_px.0 >= 0 && first_px.0 < self.width as i32 && first_px.1 >= 0 && first_px.1 < self.height as i32 {
                        let idx = first_px.1 as usize * self.width + first_px.0 as usize;
                        if idx < self.pixels.len() {
                            let pixel_value = self.pixels[idx];
                            println!("    First pixel ({}, {}) value: {} (maxed: {})", 
                                     first_px.0, first_px.1, pixel_value, pixel_value == 255);
                            
                            // If pixel is maxed out, this might be a small loop - force reset
                            if pixel_value == 255 {
                                println!("    Detected maxed-out loop - forcing attractor reset");
                                self.needs_full_refresh = true;
                                // Force immediate reset by setting status to Running
                                if self.status == AttractorStatus::Displaying {
                                    self.status = AttractorStatus::Running;
                                    self.start_time = get_time();
                                }
                            }
                        }
                    }
                }
                
                // If we've been stuck for too long with no new pixels, force completion
                if self.consecutive_empty_steps >= 2000 {  // Increased threshold for state transition
                    println!("    Attractor stuck for too long - forcing completion and transition to Displaying");
                    self.status = AttractorStatus::Displaying;
                    self.display_start_time = get_time();
                    self.consecutive_empty_steps = 0;
                }
                
                // Reset counter to avoid spam
                self.consecutive_empty_steps = 0;
            }
        } else {
            self.consecutive_empty_steps = 0;
        }
    }

    fn is_interesting(a: f64, b: f64, c: f64, d: f64, paused: bool) -> (bool, f64) {
        let mut x = 0.1;
        let mut y = 0.1;
        let mut points = std::collections::HashSet::new();
        let mut last_points_count = 0;
        let mut stagnant_count = 0;
        let mut total_iterations = 0;
        let mut new_points_count = 0;
        
        // Create temporary system for next_point calculation
        let sys = Self {
            x: 0.0, y: 0.0,  // These won't be used
            a, b, c, d,
            scalex: 0.0, scaley: 0.0,  // These won't be used
            minx: 0.0, miny: 0.0,
            pixels: Vec::new(),
            changed_pixel_indices: Vec::new(),
            needs_full_refresh: false,
            symmetry_buffer: Vec::new(),
            width: 0, height: 0,
            nonzero_pixels: 0,
            maxed_pixels: 0,
            invert: false,
            status: AttractorStatus::Running,
            display_start_time: 0.0,
            fade_start_time: 0.0,
            channel: ColorChannel::Red,
            active: true,
            start_time: 0.0,
            monochrome: false,
            color_state: ColorState::RGB,
            symmetry: SymmetryType::None,
            consecutive_empty_steps: 0,
        };
        
        // Reduced warmup phase
        for _ in 0..500 {
            let (new_x, new_y) = sys.next_point(x, y);
            x = new_x;
            y = new_y;
        }
        
        // Track the bounding box of points to measure spread
        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;
        
        // Reduced evaluation iterations
        for i in 0..3000 {
            let (new_x, new_y) = sys.next_point(x, y);
            x = new_x;
            y = new_y;
            total_iterations += 1;
            
            // Update bounding box
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            
            // Add point to set (scaled to integer coordinates for uniqueness check)
            let px = (x * 500.0) as i32;  // Reduced precision for faster HashSet operations
            let py = (y * 500.0) as i32;
            let point_was_new = points.insert((px, py));
            if point_was_new {
                new_points_count += 1;
            }
            
            // Check for stagnation every 200 iterations (less frequent)
            if i % 200 == 0 {
                if points.len() == last_points_count {
                    stagnant_count += 1;
                    // If we've been stagnant for too long, reject this attractor
                    if stagnant_count > 3 {  // Reduced threshold
                        return (false, 0.0);
                    }
                } else {
                    stagnant_count = 0;
                }
                last_points_count = points.len();
            }
            
            // Early success if we have enough unique points and good diversity
            if points.len() > 1500 {  // Reduced threshold
                // Calculate the area of the bounding box
                let area = (max_x - min_x) * (max_y - min_y);
                // Calculate percentage of new points
                let new_points_percentage = new_points_count as f64 / total_iterations as f64;
                
                // Ensure the points are well spread out (not clustered) and have good diversity
                if area > 0.5 && new_points_percentage > 0.8 {  // Relaxed criteria
                    return (true, new_points_percentage);
                }
            }
        }
        
        // For final evaluation, check point count, spread, and diversity
        let area = (max_x - min_x) * (max_y - min_y);
        let new_points_percentage = new_points_count as f64 / total_iterations as f64;
        
        // Print diversity information for tuning (only when not paused)
        if !paused {
            println!("  Diversity check: {:.1}% new points ({}/{}), {} unique points, area: {:.2}", 
                new_points_percentage * 100.0, new_points_count, total_iterations, points.len(), area);
        }
        
        // Relaxed requirements for faster generation
        let is_good = points.len() > 1000 && area > 0.5 && new_points_percentage > 0.8;
        (is_good, new_points_percentage)
    }

    fn generate_interesting_params(paused: bool) -> (f64, f64, f64, f64) {
        let mut attempts = 0;
        loop {
            // Generate random parameters
            let a = rand::gen_range(-2.0, 2.0) as f64;
            let b = rand::gen_range(-2.0, 2.0) as f64;
            let c = rand::gen_range(-2.0, 2.0) as f64;
            let d = rand::gen_range(-2.0, 2.0) as f64;
            
            // Test if these parameters generate an interesting pattern
            let (is_interesting, diversity_percentage) = Self::is_interesting(a, b, c, d, paused);
            if is_interesting {
                if !paused {
                    println!("Found good parameters after {} attempts: a={}, b={}, c={}, d={} (diversity: {:.1}%)", 
                        attempts + 1, a, b, c, d, diversity_percentage * 100.0);
                }
                return (a, b, c, d);
            }
            
            attempts += 1;
            if attempts > 100 {
                if !paused {
                    println!("Warning: Could not find ideal parameters after 100 attempts, using last set");
                }
                return (a, b, c, d);
            }
        }
    }

    fn new(x: f64, y: f64, width: usize, height: usize, channel: ColorChannel, paused: bool, color_state: ColorState, invert: bool) -> Self {
        let (a, b, c, d) = Self::generate_interesting_params(paused);
        let current_time = get_time();
        
        let mut system = Self {
            x,
            y,
            a,
            b,
            c,
            d,
            scalex: SCALE,
            scaley: SCALE,
            minx: 0.0,
            miny: 0.0,
            pixels: vec![0; width * height],
            changed_pixel_indices: Vec::new(),
            needs_full_refresh: false,
            symmetry_buffer: Vec::new(),
            width,
            height,
            nonzero_pixels: 0,
            maxed_pixels: 0,
            invert,
            status: AttractorStatus::Running,
            display_start_time: current_time,
            fade_start_time: current_time,
            channel,
            active: true,  // Start active by default
            start_time: current_time,
            monochrome: false,
            color_state,
            symmetry: SymmetryType::None, // Default to no symmetry
            consecutive_empty_steps: 0,
        };
        
        system
    }

    fn calculate_scales(&mut self) {
        let mut minx = f64::MAX;
        let mut maxx = f64::MIN;
        let mut miny = f64::MAX;
        let mut maxy = f64::MIN;
        
        self.x = 0.1;
        self.y = 0.1;
        
        // First run some iterations without tracking to let the attractor stabilize
        for _ in 0..1000 {
            let (new_x, new_y) = self.next_point(self.x, self.y);
            self.x = new_x;
            self.y = new_y;
        }
        
        // Now track the bounds for the next set of iterations
        for _ in 0..10000 {
            let (new_x, new_y) = self.next_point(self.x, self.y);
            self.x = new_x;
            self.y = new_y;
            
            minx = minx.min(self.x);
            maxx = maxx.max(self.x);
            miny = miny.min(self.y);
            maxy = maxy.max(self.y);
        }
        
        self.x = 0.1;
        self.y = 0.1;
        
        // Ensure we have a minimum range to prevent extreme scaling
        let min_range = 0.1;
        let x_range = (maxx - minx).max(min_range);
        let y_range = (maxy - miny).max(min_range);
        
        // Add margin to the bounds
        let margin = 0.1;  // 10% margin
        
        if self.symmetry == SymmetryType::None {
            // No symmetry: use independent scaling to fill screen efficiently
            let x_margin = x_range * margin;
            let y_margin = y_range * margin;
            
            self.minx = minx - x_margin;
            let maxx = maxx + x_margin;
            self.miny = miny - y_margin;
            let maxy = maxy + y_margin;
            
            // Scale each axis independently to fill the window
            self.scalex = self.width as f64 / (maxx - self.minx);
            self.scaley = self.height as f64 / (maxy - self.miny);
            
            println!("No symmetry - X range: {} to {}, Scale: {}", self.minx, maxx, self.scalex);
            println!("No symmetry - Y range: {} to {}, Scale: {}", self.miny, maxy, self.scaley);
        } else {
            // Symmetry active: use largest square that fits in the window
            let max_range = x_range.max(y_range);
            
            // Find the largest square that fits in the window
            let square_size = (self.width.min(self.height) as f64) * 0.95; // 95% of smaller dimension
            
            // Calculate scale to fit the pattern in this square
            let total_range = max_range * (1.0 + 2.0 * margin);
            let uniform_scale = square_size / total_range;
            
            // Center the pattern in the mathematical coordinate space
            let x_center = (minx + maxx) / 2.0;
            let y_center = (miny + maxy) / 2.0;
            
            // Create square bounds centered on the pattern
            let half_range = total_range / 2.0;
            self.minx = x_center - half_range;
            let maxx = x_center + half_range;
            self.miny = y_center - half_range;
            let maxy = y_center + half_range;
            
            // Use the same uniform scale for both axes
            self.scalex = uniform_scale;
            self.scaley = uniform_scale;
            
            println!("Symmetry active - Screen: {}x{}, Square size: {:.2}, Pattern range: {:.2}, Scale: {:.4}", 
                     self.width, self.height, square_size, total_range, uniform_scale);
            println!("X range: {:.2} to {:.2}, Y range: {:.2} to {:.2}", self.minx, maxx, self.miny, maxy);
        }
    }

    fn print_stats(&self) {
       if self.active {
        println!(
            "Pixels: {} active ({:.1}%), {} maxed ({:.1}%)", 
            self.nonzero_pixels,
            (self.nonzero_pixels as f64 * 100.0) / (self.width * self.height) as f64,
            self.maxed_pixels,
            (self.maxed_pixels as f64 * 100.0) / self.nonzero_pixels as f64);
        match self.status {
            AttractorStatus::Displaying => println!("Displaying for {} seconds", self.display_start_time),
            AttractorStatus::Fading => println!("Fading for {} seconds", self.fade_start_time),
            _ => {}
           }
       }
    }

    fn reset_with_random_params(&mut self, paused: bool) {
        let (a, b, c, d) = match self.color_state {
            ColorState::Correlated => {
                // For correlated mode, generate new shared parameters
                Self::generate_interesting_params(paused)
            }
            _ => {
                // For RGB and Monochrome modes, generate independent parameters
                Self::generate_interesting_params(paused)
            }
        };
        
        self.a = a;
        self.b = b;
        self.c = c;
        self.d = d;
        
        // Reset position
        self.x = 0.1;
        self.y = 0.1;
        
        // Clear pixel data
        self.pixels.fill(0);
        self.changed_pixel_indices.clear();
        self.nonzero_pixels = 0;
        self.maxed_pixels = 0;
        let current_time = get_time();
        self.display_start_time = current_time;
        self.fade_start_time = current_time;
        self.start_time = current_time;
        
        // Ensure the rendering system knows to handle the current invert state properly
        self.needs_full_refresh = true;
        
        // Recalculate scales for new parameters
        self.calculate_scales();
        self.status = AttractorStatus::Running;
    }

    fn fade_pixels(&mut self) {
        for &idx in &self.changed_pixel_indices {
            if self.pixels[idx] > 0 {
                self.pixels[idx] = self.pixels[idx].saturating_sub(4);
            }
        }
        self.changed_pixel_indices.clear();
    }

    fn mark_all_pixels_changed(&mut self) {
        self.changed_pixel_indices.clear();
        self.needs_full_refresh = true;
    }

    fn clear_full_refresh_flag(&mut self) {
        self.needs_full_refresh = false;
    }

    fn should_reset(&self, frame_count: u32) -> bool {
        // Don't reset during initial development phase
        if frame_count < 100 {
            return false;
        }

        // Reset when more than SATURATION_THRESHOLD of active pixels are maxed out
        let saturation_ratio = self.maxed_pixels as f64 / self.nonzero_pixels as f64;
        let saturated = saturation_ratio > SATURATION_THRESHOLD;
        
        // Reset if the attractor has been running for more than the maximum time
        let time_exceeded = get_time() - self.start_time > MAX_ATTRACTOR_RUNTIME;
        
        if saturated {
            println!("  Resetting due to saturation: {:.1}%", saturation_ratio * 100.0);
        } else if time_exceeded {
            println!("  Resetting due to time limit");
        }
        
        saturated || time_exceeded
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Pickover Attractor".to_owned(),
        window_width: 1280,
        window_height: 720,
        ..Default::default()
    }
}

fn draw_command_summary(inverted: bool) {
    let text_color = if inverted { BLACK } else { WHITE };
    let y_start = 10.0;
    let line_height = 20.0;
    let x_pos = 10.0;

    draw_text_improved("Pickover Attractor Controls:", x_pos, y_start, 20.0, text_color);
    draw_text_improved("Space - Reset all attractors", x_pos, y_start + line_height, 20.0, text_color);
    draw_text_improved("P     - Pause/unpause", x_pos, y_start + line_height * 2.0, 20.0, text_color);
    draw_text_improved("I     - Toggle inversion", x_pos, y_start + line_height * 3.0, 20.0, text_color);
    draw_text_improved("R     - Toggle red channel", x_pos, y_start + line_height * 4.0, 20.0, text_color);
    draw_text_improved("G     - Toggle green channel", x_pos, y_start + line_height * 5.0, 20.0, text_color);
    draw_text_improved("B     - Toggle blue channel", x_pos, y_start + line_height * 6.0, 20.0, text_color);
    draw_text_improved("M     - Toggle color state", x_pos, y_start + line_height * 7.0, 20.0, text_color);
    draw_text_improved("S     - Toggle symmetry mode", x_pos, y_start + line_height * 8.0, 20.0, text_color);
    draw_text_improved("/     - Toggle help display", x_pos, y_start + line_height * 9.0, 20.0, text_color);
    draw_text_improved("Q     - Quit program", x_pos, y_start + line_height * 10.0, 20.0, text_color);
}

fn seed_rng() {
    // Create a robust seed using WASM-compatible entropy sources
    let mut seed = 0u64;
    
    // Source 1: High-precision time (microseconds) - works in both native and WASM
    let time_seed = (get_time() * 1_000_000.0) as u64;
    seed = seed.wrapping_add(time_seed);
    
    // Source 2: Screen dimensions (provides some variation)
    let screen_seed = (screen_width() as u64).wrapping_mul(screen_height() as u64);
    seed = seed.wrapping_add(screen_seed);
    
    // Source 3: Add entropy from frame count and additional time components
    // Use macroquad's get_time() multiple times with small delays for additional entropy
    let time1 = get_time();
    let time2 = get_time();
    let time_diff = ((time2 - time1) * 1_000_000_000.0) as u64; // nanoseconds
    seed = seed.wrapping_add(time_diff);
    
    // Source 4: Add some pseudo-random variation based on time components
    let time_ms = (get_time() * 1000.0) as u64;
    seed = seed.wrapping_add(time_ms.wrapping_mul(0x5bd1e995));
    
    // Mix the seed using a simple but effective algorithm
    seed = seed.wrapping_mul(0x5bd1e995);
    seed = seed.wrapping_add(0x5bd1e995);
    seed ^= seed >> 15;
    seed = seed.wrapping_mul(0x5bd1e995);
    
    println!("RNG seeded with: {}", seed);
    rand::srand(seed);
}

// Helper function for better text rendering
fn draw_text_improved(text: &str, x: f32, y: f32, font_size: f32, color: Color) {
    // Draw text without offset to avoid rendering artifacts
    draw_text(text, x, y, font_size, color);
}

// Helper function to get display name for symmetry types
fn get_symmetry_display_name(symmetry: SymmetryType) -> &'static str {
    match symmetry {
        SymmetryType::None => "Symmetry",
        SymmetryType::Radial4 => "4-Fold",
        SymmetryType::Radial6 => "6-Fold",
        SymmetryType::Radial8 => "8-Fold",
    }
}

// Helper function to generate random correlation percentage (0-1%)
fn generate_random_correlation() -> f64 {
    rand::gen_range(0.0, 1.0) / 100.0  // Convert to percentage (0-0.01)
}



#[macroquad::main(window_conf)]
async fn main() {
    seed_rng();  // Seed the RNG with current timestamp
    
    // Initialize symmetry lookup tables for performance
    unsafe {
        SYMMETRY_TABLES = Some(SymmetryLookupTables::new());
    }
    
    // Note: Font loading is commented out due to macroquad version compatibility
    // If you want to use custom fonts, you may need to update macroquad or use a different approach
    // For now, we'll use the default font which should still look good
    println!("Using default font - for custom fonts, update macroquad version");
    
    let w = screen_width() as usize;
    let h = screen_height() as usize;
    let attractor_h = (h as f32 - UI_AREA_HEIGHT) as usize;  // Reserve space for UI
    let mut paused = false;
    let mut show_help = false;
    let mut monochrome = false;  // Global monochrome mode flag
    let mut color_state = ColorState::Monochrome;  // Current color state
    let mut shared_params = (0.0, 0.0, 0.0, 0.0);  // Shared parameters for correlated mode
    // Removed correlated_deviation slider - now using random values between 0-1%
    let mut symmetry = SymmetryType::None;  // Current symmetry mode
    let mut global_invert = false;  // Global day/night mode (false = day, true = night)

    let mut attractors = vec![
        PickoverSystem::new(0.1, 0.1, w, attractor_h, ColorChannel::Red, paused, color_state, global_invert),
        PickoverSystem::new(0.1, 0.1, w, attractor_h, ColorChannel::Green, paused, color_state, global_invert),
        PickoverSystem::new(0.1, 0.1, w, attractor_h, ColorChannel::Blue, paused, color_state, global_invert),
    ];

    // Initialize monochrome flag and symmetry for all attractors
    for attractor in attractors.iter_mut() {
        attractor.monochrome = monochrome;
        attractor.symmetry = symmetry;
    }

    let mut frame_count = 0;
    let target_frame_time = 1.0 / 60.0;
    let mut last_fps_time = get_time();
    let mut fps = 0.0;
    let mut iteration_time = 0.0;
    let mut pixel_update_time = 0.0;
    let mut iterations = 0;
    let mut prev_w = w;
    let mut prev_h = h;

    // Initialize image and texture
    let mut image = Image::gen_image_color(w as u16, attractor_h as u16, BLACK);
    let mut texture = Texture2D::from_image(&image);
    let mut image_buffer = vec![[0u8; 4]; w * attractor_h];

    // Initialize the image buffer with black background
    for pixel in image_buffer.iter_mut() {
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
        pixel[3] = 255;
    }

    // Force initial calculation of scales for all attractors
    for attractor in attractors.iter_mut() {
        attractor.calculate_scales();
        attractor.x = 0.1;
        attractor.y = 0.1;
        attractor.status = AttractorStatus::Running;
    }

    loop {
        // Handle window resize
        let current_w = screen_width() as usize;
        let current_h = screen_height() as usize;
        let current_attractor_h = (current_h as f32 - UI_AREA_HEIGHT) as usize;
        
        if current_w != prev_w || current_h != prev_h {
            println!("Window resized: {}x{} -> {}x{}", prev_w, prev_h, current_w, current_h);
            
            // Update dimensions
            let w = current_w;
            let h = current_h;
            let attractor_h = current_attractor_h;
            prev_w = w;
            prev_h = h;
            
            // Recreate image and texture
            image = Image::gen_image_color(w as u16, attractor_h as u16, BLACK);
            texture = Texture2D::from_image(&image);
            image_buffer = vec![[0u8; 4]; w * attractor_h];
            
            // Initialize the image buffer with black background
            for pixel in image_buffer.iter_mut() {
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = 0;
                pixel[3] = 255;
            }
            
            // Recreate attractors with new dimensions but preserve parameters
            let old_attractors = attractors.clone();
            attractors = vec![
                PickoverSystem::new(0.1, 0.1, w, attractor_h, ColorChannel::Red, paused, color_state, global_invert),
                PickoverSystem::new(0.1, 0.1, w, attractor_h, ColorChannel::Green, paused, color_state, global_invert),
                PickoverSystem::new(0.1, 0.1, w, attractor_h, ColorChannel::Blue, paused, color_state, global_invert),
            ];
            
            // Copy parameters from old attractors to new ones
            for (new_attractor, old_attractor) in attractors.iter_mut().zip(old_attractors.iter()) {
                new_attractor.a = old_attractor.a;
                new_attractor.b = old_attractor.b;
                new_attractor.c = old_attractor.c;
                new_attractor.d = old_attractor.d;
                new_attractor.invert = old_attractor.invert;
                new_attractor.active = old_attractor.active;
                new_attractor.symmetry = symmetry; // Use current global symmetry mode
            }
            
            // Initialize monochrome flag for all attractors
            for attractor in attractors.iter_mut() {
                attractor.monochrome = monochrome;
            }
            
            // Handle state-specific logic for color modes
            match color_state {
                ColorState::Monochrome => {
                    // Only red channel active in monochrome
                    attractors[0].active = true;
                    attractors[1].active = false;
                    attractors[2].active = false;
                    
                    // Clear pixel data from inactive attractors
                    attractors[1].pixels.fill(0);
                    attractors[1].changed_pixel_indices.clear();
                    attractors[1].nonzero_pixels = 0;
                    attractors[1].maxed_pixels = 0;
                    
                    attractors[2].pixels.fill(0);
                    attractors[2].changed_pixel_indices.clear();
                    attractors[2].nonzero_pixels = 0;
                    attractors[2].maxed_pixels = 0;
                }
                ColorState::Correlated => {
                    // All channels active with shared parameters (already preserved above)
                    attractors[0].active = true;
                    attractors[1].active = true;
                    attractors[2].active = true;
                }
                ColorState::RGB => {
                    // All channels active with independent parameters (already preserved above)
                    attractors[0].active = true;
                    attractors[1].active = true;
                    attractors[2].active = true;
                }
            }
            
            // Force recalculation of scales for new dimensions with preserved parameters
            for attractor in attractors.iter_mut() {
                attractor.calculate_scales();
                attractor.x = 0.1;
                attractor.y = 0.1;
                attractor.status = AttractorStatus::Running;
                // Clear pixel data and reset counters
                attractor.pixels.fill(0);
                attractor.changed_pixel_indices.clear();
                attractor.nonzero_pixels = 0;
                attractor.maxed_pixels = 0;
                attractor.start_time = get_time();
                attractor.display_start_time = get_time();
                attractor.fade_start_time = get_time();
            }
            
            // Clear the screen
            clear_background(if attractors[0].invert { WHITE } else { BLACK });
        }

        // Handle input
        if is_key_pressed(KeyCode::Q) {
            std::process::exit(0);
        }

        if is_key_pressed(KeyCode::Space) {
            match color_state {
                ColorState::Correlated => {
                    // Generate shared parameters for correlated mode
                    shared_params = PickoverSystem::generate_interesting_params(paused);
                    // Use the same starting position for all attractors
                    let start_x = 0.1;
                    let start_y = 0.1;
                    // Generate random correlation for this reset
                    let correlation = generate_random_correlation();
                    
                    // Apply shared parameters with deviations and same starting positions to all attractors
                    for attractor in attractors.iter_mut() {
                        attractor.a = PickoverSystem::apply_deviation(shared_params.0, correlation);
                        attractor.b = PickoverSystem::apply_deviation(shared_params.1, correlation);
                        attractor.c = PickoverSystem::apply_deviation(shared_params.2, correlation);
                        attractor.d = PickoverSystem::apply_deviation(shared_params.3, correlation);
                        attractor.x = start_x;
                        attractor.y = start_y;
                        
                        // Clear pixel data and reset
                        attractor.pixels.fill(0);
                        attractor.changed_pixel_indices.clear();
                        attractor.nonzero_pixels = 0;
                        attractor.maxed_pixels = 0;
                        attractor.status = AttractorStatus::Running;
                        attractor.start_time = get_time();
                        attractor.display_start_time = get_time();
                        attractor.fade_start_time = get_time();
                        
                        // Recalculate scales for new parameters
                        attractor.calculate_scales();
                    }
                }
                ColorState::Monochrome => {
                    // Only reset the red attractor in monochrome mode
                    attractors[0].reset_with_random_params(paused);
                    // Ensure invert state matches global setting
                    attractors[0].invert = global_invert;
                    // Clear pixel data from inactive attractors
                    attractors[1].pixels.fill(0);
                    attractors[1].changed_pixel_indices.clear();
                    attractors[1].nonzero_pixels = 0;
                    attractors[1].maxed_pixels = 0;
                    
                    attractors[2].pixels.fill(0);
                    attractors[2].changed_pixel_indices.clear();
                    attractors[2].nonzero_pixels = 0;
                    attractors[2].maxed_pixels = 0;
                }
                ColorState::RGB => {
                    // Reset all attractors with independent parameters
                    for attractor in attractors.iter_mut() {
                        attractor.reset_with_random_params(paused);
                        // Ensure invert state matches global setting
                        attractor.invert = global_invert;
                    }
                }
            }
            // Clear background and image buffer using global_invert setting
            let fill_value = if global_invert { 255 } else { 0 };
            clear_background(if global_invert { WHITE } else { BLACK });
            image_buffer.fill([fill_value, fill_value, fill_value, 255]);
            
            // Ensure all attractors have correct invert state and need refresh
            for attractor in attractors.iter_mut() {
                attractor.invert = global_invert;
                attractor.needs_full_refresh = true;
                // Force attractors stuck in "Displaying" mode back to "Running"
                if attractor.status == AttractorStatus::Displaying {
                    attractor.status = AttractorStatus::Running;
                    attractor.start_time = get_time();
                }
                attractor.consecutive_empty_steps = 0;
            }
        }

        if is_key_pressed(KeyCode::I) {
            global_invert = !global_invert;
            for attractor in attractors.iter_mut() {
                attractor.invert = global_invert;
                // Clear all pixel data when switching modes to prevent artifacts
                attractor.pixels.fill(0);
                attractor.changed_pixel_indices.clear();
                attractor.nonzero_pixels = 0;
                attractor.maxed_pixels = 0;
                // Force all pixels to be re-rendered with new inversion setting
                attractor.mark_all_pixels_changed();
            }
            let fill_value = if global_invert { 255 } else { 0 };
            clear_background(if global_invert { WHITE } else { BLACK });
            // Clear the image buffer to match the new mode
            for pixel in image_buffer.iter_mut() {
                pixel[0] = fill_value;
                pixel[1] = fill_value;
                pixel[2] = fill_value;
                pixel[3] = 255;
            }
        }

        if is_key_pressed(KeyCode::P) {
           paused = !paused;
        }

        // Add new key handlers for R, G, B (only active in RGB mode)
        if color_state == ColorState::RGB {
            if is_key_pressed(KeyCode::R) {
                attractors[0].active = !attractors[0].active;
                if !attractors[0].active {
                    // Clear the red channel
                    for pixel in image_buffer.iter_mut() {
                        pixel[0] = if attractors[0].invert { 255 } else { 0 };
                    }
                } else {
                    // Reset the attractor when toggled back on
                    attractors[0].reset_with_random_params(paused);
                    // Ensure invert state matches global setting
                    attractors[0].invert = global_invert;
                }
            }

            if is_key_pressed(KeyCode::G) {
                attractors[1].active = !attractors[1].active;
                if !attractors[1].active {
                    // Clear the green channel
                    for pixel in image_buffer.iter_mut() {
                        pixel[1] = if attractors[1].invert { 255 } else { 0 };
                    }
                } else {
                    // Reset the attractor when toggled back on
                    attractors[1].reset_with_random_params(paused);
                    // Ensure invert state matches global setting
                    attractors[1].invert = global_invert;
                }
            }

            if is_key_pressed(KeyCode::B) {
                attractors[2].active = !attractors[2].active;
                if !attractors[2].active {
                    // Clear the blue channel
                    for pixel in image_buffer.iter_mut() {
                        pixel[2] = if attractors[2].invert { 255 } else { 0 };
                    }
                } else {
                    // Reset the attractor when toggled back on
                    attractors[2].reset_with_random_params(paused);
                    // Ensure invert state matches global setting
                    attractors[2].invert = global_invert;
                }
            }
        }

        // Add monochrome mode toggle
        if is_key_pressed(KeyCode::M) {
            color_state = match color_state {
                ColorState::RGB => ColorState::Monochrome,
                ColorState::Monochrome => ColorState::Correlated,
                ColorState::Correlated => ColorState::RGB,
            };
            
            println!("Color state: {:?}", color_state);
            
            // Update color state for all attractors
            for attractor in attractors.iter_mut() {
                attractor.color_state = color_state;
            }
            
            // Handle state-specific logic
            match color_state {
                ColorState::Monochrome => {
                    // Only red channel active in monochrome
                    attractors[0].active = true;
                    attractors[1].active = false;
                    attractors[2].active = false;
                    
                    // Clear pixel data from inactive attractors
                    attractors[1].pixels.fill(0);
                    attractors[1].changed_pixel_indices.clear();
                    attractors[1].nonzero_pixels = 0;
                    attractors[1].maxed_pixels = 0;
                    
                    attractors[2].pixels.fill(0);
                    attractors[2].changed_pixel_indices.clear();
                    attractors[2].nonzero_pixels = 0;
                    attractors[2].maxed_pixels = 0;
                    
                    // Reset the red attractor with new parameters
                    attractors[0].reset_with_random_params(paused);
                    // Ensure invert state matches global setting
                    attractors[0].invert = global_invert;
                }
                ColorState::Correlated => {
                    // Generate shared parameters for correlated mode
                    shared_params = PickoverSystem::generate_interesting_params(paused);
                    // All channels active with shared parameters
                    attractors[0].active = true;
                    attractors[1].active = true;
                    attractors[2].active = true;
                    
                    // Use the same starting position for all attractors
                    let start_x = 0.1;
                    let start_y = 0.1;
                    // Generate random correlation for this mode switch
                    let correlation = generate_random_correlation();
                    
                    // Apply shared parameters with deviations and same starting positions to all attractors
                    for attractor in attractors.iter_mut() {
                        attractor.a = PickoverSystem::apply_deviation(shared_params.0, correlation);
                        attractor.b = PickoverSystem::apply_deviation(shared_params.1, correlation);
                        attractor.c = PickoverSystem::apply_deviation(shared_params.2, correlation);
                        attractor.d = PickoverSystem::apply_deviation(shared_params.3, correlation);
                        attractor.x = start_x;
                        attractor.y = start_y;
                        
                        // Clear pixel data and reset
                        attractor.pixels.fill(0);
                        attractor.changed_pixel_indices.clear();
                        attractor.nonzero_pixels = 0;
                        attractor.maxed_pixels = 0;
                        attractor.status = AttractorStatus::Running;
                        attractor.start_time = get_time();
                        attractor.display_start_time = get_time();
                        attractor.fade_start_time = get_time();
                        
                        // Recalculate scales for new parameters
                        attractor.calculate_scales();
                    }
                }
                ColorState::RGB => {
                    // All channels active with independent parameters
                    attractors[0].active = true;
                    attractors[1].active = true;
                    attractors[2].active = true;
                    // Reset all attractors with independent parameters
                    for attractor in attractors.iter_mut() {
                        attractor.reset_with_random_params(paused);
                        // Ensure invert state matches global setting
                        attractor.invert = global_invert;
                    }
                }
            }
            
            // Clear the screen
            clear_background(if attractors[0].invert { WHITE } else { BLACK });
            let fill_value = if attractors[0].invert { 255 } else { 0 };
            for pixel in image_buffer.iter_mut() {
                pixel[0] = fill_value;
                pixel[1] = fill_value;
                pixel[2] = fill_value;
                pixel[3] = 255;
            }
        }

        // Add symmetry mode toggle
        if is_key_pressed(KeyCode::S) {
            symmetry = match symmetry {
                SymmetryType::None => SymmetryType::Radial4,
                SymmetryType::Radial4 => SymmetryType::Radial6,
                SymmetryType::Radial6 => SymmetryType::Radial8,
                SymmetryType::Radial8 => SymmetryType::None,
            };
            
            println!("Symmetry mode: {:?}", symmetry);
            
            // Update symmetry for all attractors and recalculate scales
            for attractor in attractors.iter_mut() {
                attractor.symmetry = symmetry;
                attractor.calculate_scales();
                
                // Clear pixel data but keep the same parameters
                attractor.pixels.fill(0);
                attractor.changed_pixel_indices.clear();
                attractor.nonzero_pixels = 0;
                attractor.maxed_pixels = 0;
                attractor.status = AttractorStatus::Running;
                attractor.start_time = get_time();
                attractor.display_start_time = get_time();
                attractor.fade_start_time = get_time();
                
                // Reset position to starting point but keep parameters
                attractor.x = 0.1;
                attractor.y = 0.1;
            }
            
            // Clear the screen
            clear_background(if attractors[0].invert { WHITE } else { BLACK });
            let fill_value = if attractors[0].invert { 255 } else { 0 };
            for pixel in image_buffer.iter_mut() {
                pixel[0] = fill_value;
                pixel[1] = fill_value;
                pixel[2] = fill_value;
                pixel[3] = 255;
            }
        }

        if !paused {
            iterations = 0;
            let mut needs_correlated_reset = false;
            
            // In monochrome mode, only process the red attractor
            let attractors_to_process = if color_state == ColorState::Monochrome {
                &mut attractors[0..1]
            } else {
                &mut attractors[..]
            };
            
            for attractor in attractors_to_process {
                if !attractor.active {
                    continue;
                }

                match attractor.status {
                    AttractorStatus::Running => {
                        let iteration_start = get_time();
                        let mut local_iterations = 0;
                        while (get_time() - iteration_start) < target_frame_time * 0.6 {
                            attractor.step();
                            local_iterations += 1;
                        }
                        iterations += local_iterations;
                        iteration_time = get_time() - iteration_start;

                        if attractor.should_reset(frame_count) {
                            // In correlated mode, immediately mark for reset of all attractors
                            if color_state == ColorState::Correlated {
                                needs_correlated_reset = true;
                            }
                            println!("  Attractor transitioning to Displaying state (frame {})", frame_count);
                            attractor.status = AttractorStatus::Displaying;
                            attractor.display_start_time = get_time();
                        }
                    }
                    AttractorStatus::Displaying => {
                        let display_elapsed = get_time() - attractor.display_start_time;
                        if display_elapsed >= DISPLAY_DURATION {
                            println!("  Attractor transitioning to Fading state after {:.1}s", display_elapsed);
                            attractor.status = AttractorStatus::Fading;
                            attractor.fade_start_time = get_time();
                        } else if display_elapsed > DISPLAY_DURATION * 2.0 {
                            // Safety mechanism: force transition if stuck in displaying for too long
                            println!("  Safety reset: attractor stuck in displaying for {:.1}s, forcing fade", display_elapsed);
                            attractor.status = AttractorStatus::Fading;
                            attractor.fade_start_time = get_time();
                        }
                    }
                    AttractorStatus::Fading => {
                        let fade_elapsed = get_time() - attractor.fade_start_time;
                        if fade_elapsed >= FADE_DURATION {
                            // Reset based on current color state
                            match color_state {
                                ColorState::Correlated => {
                                    // In correlated mode, the flag was already set when should_reset() returned true
                                    // Just transition to running state - the reset will be handled after the loop
                                    attractor.status = AttractorStatus::Running;
                                }
                                _ => {
                                    // For RGB and Monochrome modes, use individual reset
                                    attractor.reset_with_random_params(paused);
                                    // Ensure invert state matches global setting
                                    attractor.invert = global_invert;
                                    attractor.status = AttractorStatus::Running;
                                }
                            }
                            
                            // Clear the appropriate channel in the image buffer
                            let fill_value = if attractor.invert { 255 } else { 0 };
                            match (color_state, attractor.channel) {
                                (ColorState::Monochrome, ColorChannel::Red) => {
                                    // In monochrome mode, clear all RGB channels with the same value
                                    image_buffer.iter_mut().for_each(|p| {
                                        p[0] = fill_value;
                                        p[1] = fill_value;
                                        p[2] = fill_value;
                                    });
                                }
                                (_, ColorChannel::Red) => image_buffer.iter_mut().for_each(|p| p[0] = fill_value),
                                (_, ColorChannel::Green) => image_buffer.iter_mut().for_each(|p| p[1] = fill_value),
                                (_, ColorChannel::Blue) => image_buffer.iter_mut().for_each(|p| p[2] = fill_value),
                            }
                        } else {
                            // Fade every other frame to control fade speed
                            // Skip individual attractor fade in monochrome mode - handled separately
                            let should_skip_fade = color_state == ColorState::Monochrome && matches!(attractor.channel, ColorChannel::Red);
                            if frame_count % 2 == 0 && !should_skip_fade {
                                attractor.fade_pixels();
                            }
                        }
                    }
                }
            }
            
            // Handle correlated reset after the loop to avoid borrowing issues
            if needs_correlated_reset {
                // Generate new shared parameters for correlated mode
                shared_params = PickoverSystem::generate_interesting_params(paused);
                // Use the same starting position for all attractors
                let start_x = 0.1;
                let start_y = 0.1;
                // Generate random correlation for this reset
                let correlation = generate_random_correlation();
                
                // Clear the entire image buffer since all attractors are being reset
                // But respect the current invert state and any pending full refresh needs
                let fill_value = if attractors[0].invert { 255 } else { 0 };
                image_buffer.iter_mut().for_each(|p| {
                    p[0] = fill_value;
                    p[1] = fill_value;
                    p[2] = fill_value;
                });
                
                // If any attractor needs a full refresh (e.g., after inversion), 
                // apply the inversion to the newly cleared buffer
                for attractor in attractors.iter_mut() {
                    if attractor.needs_full_refresh {
                        // Apply inversion to the entire image buffer
                        for pixel in image_buffer.iter_mut() {
                            pixel[0] = 255 - pixel[0];
                            pixel[1] = 255 - pixel[1];
                            pixel[2] = 255 - pixel[2];
                        }
                        // Clear the flag since we've applied the refresh
                        attractor.clear_full_refresh_flag();
                    }
                }
                
                // Apply shared parameters with deviations and same starting positions to all attractors
                for attractor in attractors.iter_mut() {
                    attractor.a = PickoverSystem::apply_deviation(shared_params.0, correlation);
                    attractor.b = PickoverSystem::apply_deviation(shared_params.1, correlation);
                    attractor.c = PickoverSystem::apply_deviation(shared_params.2, correlation);
                    attractor.d = PickoverSystem::apply_deviation(shared_params.3, correlation);
                    attractor.x = start_x;
                    attractor.y = start_y;
                    
                    // Clear pixel data and reset
                    attractor.pixels.fill(0);
                    attractor.changed_pixel_indices.clear();
                    attractor.nonzero_pixels = 0;
                    attractor.maxed_pixels = 0;
                    attractor.status = AttractorStatus::Running;
                    attractor.start_time = get_time();
                    attractor.display_start_time = get_time();
                    attractor.fade_start_time = get_time();
                    
                    // Recalculate scales for new parameters
                    attractor.calculate_scales();
                }
                
                // Reset the flag
                needs_correlated_reset = false;
            }
        }

        let pixel_start = get_time();
        let image_data = image.get_image_data_mut();

        // Update image buffer based on color state
        match color_state {
            ColorState::Monochrome => {
                // In monochrome mode, only use the red attractor and copy its values to all channels
                let red_attractor = &mut attractors[0];
                
                // During fade, don't update from attractor's pixel data - only apply unified fade
                if red_attractor.status == AttractorStatus::Fading {
                    // Apply fade to all RGB channels together in monochrome mode
                    if frame_count % 2 == 0 {
                        for pixel in image_buffer.iter_mut() {
                            if red_attractor.invert {
                                // In inverted mode, fade towards white (255)
                                if pixel[0] < 255 || pixel[1] < 255 || pixel[2] < 255 {
                                    let fade_amount = 4;
                                    pixel[0] = pixel[0].saturating_add(fade_amount);
                                    pixel[1] = pixel[1].saturating_add(fade_amount);
                                    pixel[2] = pixel[2].saturating_add(fade_amount);
                                }
                            } else {
                                // In normal mode, fade towards black (0)
                                if pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0 {
                                    let fade_amount = 4;
                                    pixel[0] = pixel[0].saturating_sub(fade_amount);
                                    pixel[1] = pixel[1].saturating_sub(fade_amount);
                                    pixel[2] = pixel[2].saturating_sub(fade_amount);
                                }
                            }
                        }
                    }
                } else {
                    // Normal operation - update from attractor's pixel data
                    if red_attractor.needs_full_refresh {
                        // Full refresh needed (e.g., after inversion)
                        for (idx, pixel) in image_buffer.iter_mut().enumerate() {
                            let intensity = red_attractor.pixels[idx];
                            let pixel_value = if red_attractor.invert { 255 - intensity } else { intensity };
                            pixel[0] = pixel_value;
                            pixel[1] = pixel_value;
                            pixel[2] = pixel_value;
                        }
                        red_attractor.clear_full_refresh_flag();
                    } else {
                        // Only update changed pixels
                        for &idx in &red_attractor.changed_pixel_indices {
                            let pixel_value = if red_attractor.invert { 255 - red_attractor.pixels[idx] } else { red_attractor.pixels[idx] };
                            // Ensure all RGB channels have the same value to maintain monochrome appearance
                            image_buffer[idx][0] = pixel_value;
                            image_buffer[idx][1] = pixel_value;
                            image_buffer[idx][2] = pixel_value;
                        }
                        red_attractor.changed_pixel_indices.clear();
                    }
                }
                
                // When red attractor starts fading, ensure all channels are in sync
                if red_attractor.status == AttractorStatus::Fading {
                    // Ensure all RGB channels have the same values before starting fade
                    for (idx, pixel) in image_buffer.iter_mut().enumerate() {
                        let current_value = pixel[0]; // Use red channel as reference
                        pixel[1] = current_value;
                        pixel[2] = current_value;
                    }
                }
            }
            ColorState::Correlated => {
                // In correlated mode, all channels use the same parameters but different starting positions
                for attractor in attractors.iter_mut() {
                    if !attractor.active {
                        continue;
                    }
                    if attractor.needs_full_refresh {
                        // Full refresh needed (e.g., after inversion)
                        for (idx, pixel) in image_buffer.iter_mut().enumerate() {
                            let intensity = attractor.pixels[idx];
                            let pixel_value = if attractor.invert { 255 - intensity } else { intensity };
                            match attractor.channel {
                                ColorChannel::Red => pixel[0] = pixel_value,
                                ColorChannel::Green => pixel[1] = pixel_value,
                                ColorChannel::Blue => pixel[2] = pixel_value,
                            }
                        }
                        attractor.clear_full_refresh_flag();
                    } else {
                        // Only update changed pixels
                        for &idx in &attractor.changed_pixel_indices {
                            let pixel_value = if attractor.invert { 255 - attractor.pixels[idx] } else { attractor.pixels[idx] };
                            match attractor.channel {
                                ColorChannel::Red => image_buffer[idx][0] = pixel_value,
                                ColorChannel::Green => image_buffer[idx][1] = pixel_value,
                                ColorChannel::Blue => image_buffer[idx][2] = pixel_value,
                            }
                        }
                        attractor.changed_pixel_indices.clear();
                    }
                }
            }
            ColorState::RGB => {
                // Normal RGB mode - update each channel independently
                for attractor in attractors.iter_mut() {
                    if !attractor.active {
                        continue;
                    }
                    if attractor.needs_full_refresh {
                        // Full refresh needed (e.g., after inversion)
                        for (idx, pixel) in image_buffer.iter_mut().enumerate() {
                            let intensity = attractor.pixels[idx];
                            let pixel_value = if attractor.invert { 255 - intensity } else { intensity };
                            match attractor.channel {
                                ColorChannel::Red => pixel[0] = pixel_value,
                                ColorChannel::Green => pixel[1] = pixel_value,
                                ColorChannel::Blue => pixel[2] = pixel_value,
                            }
                        }
                        attractor.clear_full_refresh_flag();
                    } else {
                        // Only update changed pixels
                        for &idx in &attractor.changed_pixel_indices {
                            let pixel_value = if attractor.invert { 255 - attractor.pixels[idx] } else { attractor.pixels[idx] };
                            match attractor.channel {
                                ColorChannel::Red => image_buffer[idx][0] = pixel_value,
                                ColorChannel::Green => image_buffer[idx][1] = pixel_value,
                                ColorChannel::Blue => image_buffer[idx][2] = pixel_value,
                            }
                        }
                        attractor.changed_pixel_indices.clear();
                    }
                }
            }
        }

        // Copy buffer to image
        for (pixel, buffer) in image_data.iter_mut().zip(image_buffer.iter()) {
            *pixel = *buffer;
        }

        pixel_update_time = get_time() - pixel_start;

        texture.update(&image);
        draw_texture(&texture, 0.0, 0.0, WHITE);

        // Draw UI controls at the bottom - direct drawing instead of UI windows
        let ui_y = screen_height() - UI_AREA_HEIGHT + 10.0;
        
        // Enhanced UI color palette - modern dark theme
        let text_color = Color::new(0.95, 0.95, 0.95, 1.0);  // Slightly off-white for better readability
        let label_color = Color::new(0.8, 0.8, 0.8, 1.0);    // Slightly dimmer for labels
        let button_bg_color = Color::new(0.15, 0.15, 0.18, 0.95);  // Dark blue-gray background
        let button_active_color = Color::new(0.25, 0.25, 0.35, 0.95);  // Active button color
        let button_hover_color = Color::new(0.2, 0.2, 0.25, 0.95);  // Hover state color
        
        // Improved spacing and layout
        let ui_padding = 15.0;  // Increased padding for better breathing room
        let button_height = 28.0;  // Slightly taller buttons
        let button_spacing = 8.0;  // Space between buttons
        let row_spacing = 12.0;  // Space between rows
        
        // Draw RGB button with improved layout
        let rgb_button_rect = Rect::new(ui_padding, ui_y + 10.0, 85.0, button_height);
        let rgb_button_color = if color_state == ColorState::RGB { 
            button_active_color
        } else { 
            button_bg_color
        };
        draw_rectangle(rgb_button_rect.x, rgb_button_rect.y, rgb_button_rect.w, rgb_button_rect.h, rgb_button_color);
        let rgb_button_text = if attractors[0].invert { "CYM" } else { "RGB" };
        let text_y = rgb_button_rect.y + (button_height + 16.0) / 2.0;  // Center text vertically
        draw_text_improved(rgb_button_text, rgb_button_rect.x + 8.0, text_y, 16.0, text_color);
        
        // Draw Monochrome button with proper spacing
        let mono_button_rect = Rect::new(ui_padding + 85.0 + button_spacing, ui_y + 10.0, 85.0, button_height);
        let mono_button_color = if color_state == ColorState::Monochrome { 
            button_active_color
        } else { 
            button_bg_color
        };
        draw_rectangle(mono_button_rect.x, mono_button_rect.y, mono_button_rect.w, mono_button_rect.h, mono_button_color);
        let text_y = mono_button_rect.y + (button_height + 16.0) / 2.0;  // Center text vertically
        draw_text_improved("Monochrome", mono_button_rect.x + 8.0, text_y, 16.0, text_color);
        
        // Draw Correlated button with proper spacing
        let corr_button_rect = Rect::new(ui_padding + 85.0 + 85.0 + button_spacing * 2.0, ui_y + 10.0, 85.0, button_height);
        let corr_button_color = if color_state == ColorState::Correlated { 
            button_active_color
        } else { 
            button_bg_color
        };
        draw_rectangle(corr_button_rect.x, corr_button_rect.y, corr_button_rect.w, corr_button_rect.h, corr_button_color);
        let text_y = corr_button_rect.y + (button_height + 16.0) / 2.0;  // Center text vertically
        draw_text_improved("Correlated", corr_button_rect.x + 8.0, text_y, 16.0, text_color);
        
        // Draw Symmetry button with proper spacing
        let symmetry_button_rect = Rect::new(ui_padding + 85.0 + 85.0 + 85.0 + button_spacing * 3.0, ui_y + 10.0, 75.0, button_height);
        let symmetry_button_color = if symmetry != SymmetryType::None { 
            button_active_color
        } else { 
            button_bg_color
        };
        draw_rectangle(symmetry_button_rect.x, symmetry_button_rect.y, symmetry_button_rect.w, symmetry_button_rect.h, symmetry_button_color);
        let symmetry_text = get_symmetry_display_name(symmetry);
        let text_y = symmetry_button_rect.y + (button_height + 16.0) / 2.0;  // Center text vertically
        draw_text_improved(symmetry_text, symmetry_button_rect.x + 8.0, text_y, 16.0, text_color);
        
        // Draw Day/Night button with improved layout
        let day_button_rect = Rect::new(ui_padding, ui_y + 10.0 + button_height + row_spacing, 55.0, button_height);
        let day_button_text = if attractors[0].invert { "Night" } else { "Day" };
        draw_rectangle(day_button_rect.x, day_button_rect.y, day_button_rect.w, day_button_rect.h, button_bg_color);
        let text_y = day_button_rect.y + (button_height + 16.0) / 2.0;  // Center text vertically
        draw_text_improved(day_button_text, day_button_rect.x + 8.0, text_y, 16.0, text_color);
        
        // Draw Pause/Resume button with proper spacing
        let pause_button_rect = Rect::new(ui_padding + 55.0 + button_spacing, ui_y + 10.0 + button_height + row_spacing, 65.0, button_height);
        let pause_button_text = if paused { "Resume" } else { "Pause" };
        draw_rectangle(pause_button_rect.x, pause_button_rect.y, pause_button_rect.w, pause_button_rect.h, button_bg_color);
        let text_y = pause_button_rect.y + (button_height + 16.0) / 2.0;  // Center text vertically
        draw_text_improved(pause_button_text, pause_button_rect.x + 8.0, text_y, 16.0, text_color);
        
        // Draw Help button with proper spacing
        let help_button_rect = Rect::new(ui_padding + 55.0 + 65.0 + button_spacing * 2.0, ui_y + 10.0 + button_height + row_spacing, 35.0, button_height);
        draw_rectangle(help_button_rect.x, help_button_rect.y, help_button_rect.w, help_button_rect.h, button_bg_color);
        let text_y = help_button_rect.y + (button_height + 16.0) / 2.0;  // Center text vertically
        draw_text_improved("?", help_button_rect.x + 12.0, text_y, 16.0, text_color);
        
        // Draw Next button with proper spacing
        let next_button_rect = Rect::new(ui_padding + 55.0 + 65.0 + 35.0 + button_spacing * 3.0, ui_y + 10.0 + button_height + row_spacing, 45.0, button_height);
        draw_rectangle(next_button_rect.x, next_button_rect.y, next_button_rect.w, next_button_rect.h, button_bg_color);
        let text_y = next_button_rect.y + (button_height + 16.0) / 2.0;  // Center text vertically
        draw_text_improved("Next", next_button_rect.x + 8.0, text_y, 16.0, text_color);
        
        // Draw Quit button with proper spacing
        let quit_button_rect = Rect::new(ui_padding + 55.0 + 65.0 + 35.0 + 45.0 + button_spacing * 4.0, ui_y + 10.0 + button_height + row_spacing, 45.0, button_height);
        draw_rectangle(quit_button_rect.x, quit_button_rect.y, quit_button_rect.w, quit_button_rect.h, button_bg_color);
        let text_y = quit_button_rect.y + (button_height + 16.0) / 2.0;  // Center text vertically
        draw_text_improved("Quit", quit_button_rect.x + 8.0, text_y, 16.0, text_color);
        
        // Handle mouse clicks for buttons
        if is_mouse_button_pressed(MouseButton::Left) {
            let mouse_pos = mouse_position();
            
            // Check if click is in the display area (above the UI area)
            if mouse_pos.1 < screen_height() - UI_AREA_HEIGHT {
                // Left click in display area acts as Next button
                match color_state {
                    ColorState::Correlated => {
                        // Generate shared parameters for correlated mode
                        shared_params = PickoverSystem::generate_interesting_params(paused);
                        // Use the same starting position for all attractors
                        let start_x = 0.1;
                        let start_y = 0.1;
                        // Generate random correlation for this click
                        let correlation = generate_random_correlation();
                        
                        // Apply shared parameters with deviations and same starting positions to all attractors
                        for attractor in attractors.iter_mut() {
                            attractor.a = PickoverSystem::apply_deviation(shared_params.0, correlation);
                            attractor.b = PickoverSystem::apply_deviation(shared_params.1, correlation);
                            attractor.c = PickoverSystem::apply_deviation(shared_params.2, correlation);
                            attractor.d = PickoverSystem::apply_deviation(shared_params.3, correlation);
                            attractor.x = start_x;
                            attractor.y = start_y;
                            
                            // Clear pixel data and reset
                            attractor.pixels.fill(0);
                            attractor.changed_pixel_indices.clear();
                            attractor.nonzero_pixels = 0;
                            attractor.maxed_pixels = 0;
                            attractor.status = AttractorStatus::Running;
                            attractor.start_time = get_time();
                            attractor.display_start_time = get_time();
                            attractor.fade_start_time = get_time();
                            
                            // Recalculate scales for new parameters
                            attractor.calculate_scales();
                        }
                    }
                    ColorState::Monochrome => {
                        // Only reset the red attractor in monochrome mode
                        attractors[0].reset_with_random_params(paused);
                        // Clear pixel data from inactive attractors
                        attractors[1].pixels.fill(0);
                        attractors[1].changed_pixel_indices.clear();
                        attractors[1].nonzero_pixels = 0;
                        attractors[1].maxed_pixels = 0;
                        
                        attractors[2].pixels.fill(0);
                        attractors[2].changed_pixel_indices.clear();
                        attractors[2].nonzero_pixels = 0;
                        attractors[2].maxed_pixels = 0;
                    }
                    ColorState::RGB => {
                        // Reset all attractors with independent parameters
                        for attractor in attractors.iter_mut() {
                            attractor.reset_with_random_params(paused);
                        }
                    }
                }
                
                // Clear background and image buffer respecting current day/night mode
                clear_background(if global_invert { WHITE } else { BLACK });
                let fill_value = if global_invert { 255 } else { 0 };
                image_buffer.fill([fill_value, fill_value, fill_value, 255]);
                
                // Ensure all attractors have the current global invert state
                for attractor in attractors.iter_mut() {
                    attractor.invert = global_invert;
                    // Mark for full refresh to ensure proper rendering
                    attractor.needs_full_refresh = true;
                }
            }
            
            // RGB button click
            if rgb_button_rect.contains(vec2(mouse_pos.0, mouse_pos.1)) {
                if color_state != ColorState::RGB {
                    color_state = ColorState::RGB;
                    println!("Color state: {:?}", color_state);
                    
                    // Update color state for all attractors
                    for attractor in attractors.iter_mut() {
                        attractor.color_state = color_state;
                    }
                    
                    // All channels active with independent parameters
                    attractors[0].active = true;
                    attractors[1].active = true;
                    attractors[2].active = true;
                    // Reset all attractors with independent parameters
                    for attractor in attractors.iter_mut() {
                        attractor.reset_with_random_params(paused);
                    }
                    
                    // Clear the screen respecting current day/night mode
                    clear_background(if global_invert { WHITE } else { BLACK });
                    let fill_value = if global_invert { 255 } else { 0 };
                    for pixel in image_buffer.iter_mut() {
                        pixel[0] = fill_value;
                        pixel[1] = fill_value;
                        pixel[2] = fill_value;
                        pixel[3] = 255;
                    }
                    
                    // Ensure all attractors have the current global invert state
                    for attractor in attractors.iter_mut() {
                        attractor.invert = global_invert;
                        attractor.needs_full_refresh = true;
                    }
                }
            }
            
            // Monochrome button click
            if mono_button_rect.contains(vec2(mouse_pos.0, mouse_pos.1)) {
                if color_state != ColorState::Monochrome {
                    color_state = ColorState::Monochrome;
                    println!("Color state: {:?}", color_state);
                    
                    // Update color state for all attractors
                    for attractor in attractors.iter_mut() {
                        attractor.color_state = color_state;
                    }
                    
                    // Only red channel active in monochrome
                    attractors[0].active = true;
                    attractors[1].active = false;
                    attractors[2].active = false;
                    
                    // Clear pixel data from inactive attractors
                    attractors[1].pixels.fill(0);
                    attractors[1].changed_pixel_indices.clear();
                    attractors[1].nonzero_pixels = 0;
                    attractors[1].maxed_pixels = 0;
                    
                    attractors[2].pixels.fill(0);
                    attractors[2].changed_pixel_indices.clear();
                    attractors[2].nonzero_pixels = 0;
                    attractors[2].maxed_pixels = 0;
                    
                    // Reset the red attractor with new parameters
                    attractors[0].reset_with_random_params(paused);
                    
                    // Clear the screen respecting current day/night mode
                    clear_background(if global_invert { WHITE } else { BLACK });
                    let fill_value = if global_invert { 255 } else { 0 };
                    for pixel in image_buffer.iter_mut() {
                        pixel[0] = fill_value;
                        pixel[1] = fill_value;
                        pixel[2] = fill_value;
                        pixel[3] = 255;
                    }
                    
                    // Ensure all attractors have the current global invert state
                    for attractor in attractors.iter_mut() {
                        attractor.invert = global_invert;
                        attractor.needs_full_refresh = true;
                    }
                }
            }
            
            // Correlated button click
            if corr_button_rect.contains(vec2(mouse_pos.0, mouse_pos.1)) {
                if color_state != ColorState::Correlated {
                    color_state = ColorState::Correlated;
                    println!("Color state: {:?}", color_state);
                    
                    // Update color state for all attractors
                    for attractor in attractors.iter_mut() {
                        attractor.color_state = color_state;
                    }
                    
                    // Generate shared parameters for correlated mode
                    shared_params = PickoverSystem::generate_interesting_params(paused);
                    // All channels active with shared parameters
                    attractors[0].active = true;
                    attractors[1].active = true;
                    attractors[2].active = true;
                    
                    // Use the same starting position for all attractors
                    let start_x = 0.1;
                    let start_y = 0.1;
                    // Generate random correlation for this button click
                    let correlation = generate_random_correlation();
                    
                    // Apply shared parameters with deviations and same starting positions to all attractors
                    for attractor in attractors.iter_mut() {
                        attractor.a = PickoverSystem::apply_deviation(shared_params.0, correlation);
                        attractor.b = PickoverSystem::apply_deviation(shared_params.1, correlation);
                        attractor.c = PickoverSystem::apply_deviation(shared_params.2, correlation);
                        attractor.d = PickoverSystem::apply_deviation(shared_params.3, correlation);
                        attractor.x = start_x;
                        attractor.y = start_y;
                        
                        // Clear pixel data and reset
                        attractor.pixels.fill(0);
                        attractor.changed_pixel_indices.clear();
                        attractor.nonzero_pixels = 0;
                        attractor.maxed_pixels = 0;
                        attractor.status = AttractorStatus::Running;
                        attractor.start_time = get_time();
                        attractor.display_start_time = get_time();
                        attractor.fade_start_time = get_time();
                        
                        // Recalculate scales for new parameters
                        attractor.calculate_scales();
                    }
                    
                    // Clear the screen respecting current day/night mode
                    clear_background(if global_invert { WHITE } else { BLACK });
                    let fill_value = if global_invert { 255 } else { 0 };
                    for pixel in image_buffer.iter_mut() {
                        pixel[0] = fill_value;
                        pixel[1] = fill_value;
                        pixel[2] = fill_value;
                        pixel[3] = 255;
                    }
                    
                    // Ensure all attractors have the current global invert state
                    for attractor in attractors.iter_mut() {
                        attractor.invert = global_invert;
                        attractor.needs_full_refresh = true;
                    }
                }
            }
            
            // Symmetry button click
            if symmetry_button_rect.contains(vec2(mouse_pos.0, mouse_pos.1)) {
                symmetry = match symmetry {
                    SymmetryType::None => SymmetryType::Radial4,
                    SymmetryType::Radial4 => SymmetryType::Radial6,
                    SymmetryType::Radial6 => SymmetryType::Radial8,
                    SymmetryType::Radial8 => SymmetryType::None,
                };
                
                println!("Symmetry mode: {:?}", symmetry);
                
                // Update symmetry for all attractors and recalculate scales
                for attractor in attractors.iter_mut() {
                    attractor.symmetry = symmetry;
                    attractor.calculate_scales();
                    
                    // Clear pixel data but keep the same parameters
                    attractor.pixels.fill(0);
                    attractor.changed_pixel_indices.clear();
                    attractor.nonzero_pixels = 0;
                    attractor.maxed_pixels = 0;
                    attractor.status = AttractorStatus::Running;
                    attractor.start_time = get_time();
                    attractor.display_start_time = get_time();
                    attractor.fade_start_time = get_time();
                    
                    // Reset position to starting point but keep parameters
                    attractor.x = 0.1;
                    attractor.y = 0.1;
                }
                
                // Clear the screen respecting current day/night mode
                clear_background(if global_invert { WHITE } else { BLACK });
                let fill_value = if global_invert { 255 } else { 0 };
                for pixel in image_buffer.iter_mut() {
                    pixel[0] = fill_value;
                    pixel[1] = fill_value;
                    pixel[2] = fill_value;
                    pixel[3] = 255;
                }
                
                // Ensure all attractors have the current global invert state
                for attractor in attractors.iter_mut() {
                    attractor.invert = global_invert;
                    attractor.needs_full_refresh = true;
                }
            }
            
            // Day/Night button click
            if day_button_rect.contains(vec2(mouse_pos.0, mouse_pos.1)) {
                global_invert = !global_invert;
                // Toggle inversion for all attractors
                for attractor in attractors.iter_mut() {
                    attractor.invert = global_invert;
                    // Clear all pixel data when switching modes to prevent artifacts
                    attractor.pixels.fill(0);
                    attractor.changed_pixel_indices.clear();
                    attractor.nonzero_pixels = 0;
                    attractor.maxed_pixels = 0;
                    // Force attractors in Displaying mode back to Running so they start drawing immediately
                    if attractor.status == AttractorStatus::Displaying {
                        attractor.status = AttractorStatus::Running;
                        attractor.start_time = get_time(); // Reset the timer
                    }
                    // Reset the empty steps counter
                    attractor.consecutive_empty_steps = 0;
                    // Mark for refresh
                    attractor.needs_full_refresh = true;
                }
                let fill_value = if global_invert { 255 } else { 0 };
                clear_background(if global_invert { WHITE } else { BLACK });
                // Clear the image buffer to match the new mode
                for pixel in image_buffer.iter_mut() {
                    pixel[0] = fill_value;
                    pixel[1] = fill_value;
                    pixel[2] = fill_value;
                    pixel[3] = 255;
                }
            }
            
            // Pause/Resume button click
            if pause_button_rect.contains(vec2(mouse_pos.0, mouse_pos.1)) {
                paused = !paused;
                println!("Paused: {}", paused);
            }
            
            // Help button click
            if help_button_rect.contains(vec2(mouse_pos.0, mouse_pos.1)) {
                show_help = !show_help;
                println!("Help display toggled: {}", if show_help { "shown" } else { "hidden" });
            }
            
            // Next button click
            if next_button_rect.contains(vec2(mouse_pos.0, mouse_pos.1)) {
                match color_state {
                    ColorState::Correlated => {
                        // Generate shared parameters for correlated mode
                        shared_params = PickoverSystem::generate_interesting_params(paused);
                        // Use the same starting position for all attractors
                        let start_x = 0.1;
                        let start_y = 0.1;
                        // Generate random correlation for this next click
                        let correlation = generate_random_correlation();
                        
                        // Apply shared parameters with deviations and same starting positions to all attractors
                        for attractor in attractors.iter_mut() {
                            attractor.a = PickoverSystem::apply_deviation(shared_params.0, correlation);
                            attractor.b = PickoverSystem::apply_deviation(shared_params.1, correlation);
                            attractor.c = PickoverSystem::apply_deviation(shared_params.2, correlation);
                            attractor.d = PickoverSystem::apply_deviation(shared_params.3, correlation);
                            attractor.x = start_x;
                            attractor.y = start_y;
                            
                            // Clear pixel data and reset
                            attractor.pixels.fill(0);
                            attractor.changed_pixel_indices.clear();
                            attractor.nonzero_pixels = 0;
                            attractor.maxed_pixels = 0;
                            attractor.status = AttractorStatus::Running;
                            attractor.start_time = get_time();
                            attractor.display_start_time = get_time();
                            attractor.fade_start_time = get_time();
                            
                            // Ensure invert state matches global setting
                            attractor.invert = global_invert;
                            
                            // Recalculate scales for new parameters
                            attractor.calculate_scales();
                        }
                    }
                    ColorState::Monochrome => {
                        // Only reset the red attractor in monochrome mode
                        attractors[0].reset_with_random_params(paused);
                        // Ensure invert state matches global setting
                        attractors[0].invert = global_invert;
                        // Clear pixel data from inactive attractors
                        attractors[1].pixels.fill(0);
                        attractors[1].changed_pixel_indices.clear();
                        attractors[1].nonzero_pixels = 0;
                        attractors[1].maxed_pixels = 0;
                        
                        attractors[2].pixels.fill(0);
                        attractors[2].changed_pixel_indices.clear();
                        attractors[2].nonzero_pixels = 0;
                        attractors[2].maxed_pixels = 0;
                    }
                    ColorState::RGB => {
                        // Reset all attractors with independent parameters
                        for attractor in attractors.iter_mut() {
                            attractor.reset_with_random_params(paused);
                            // Ensure invert state matches global setting
                            attractor.invert = global_invert;
                        }
                    }
                }
                // Ensure all attractors have the current global invert state
                for attractor in attractors.iter_mut() {
                    if attractor.invert != global_invert {
                        attractor.invert = global_invert;
                        // Clear all pixel data when switching modes to prevent artifacts
                        attractor.pixels.fill(0);
                        attractor.changed_pixel_indices.clear();
                        attractor.nonzero_pixels = 0;
                        attractor.maxed_pixels = 0;
                        // Force all pixels to be re-rendered with new inversion setting
                        attractor.mark_all_pixels_changed();
                    }
                }
                
                clear_background(if global_invert { WHITE } else { BLACK });
                let fill_value = if global_invert { 255 } else { 0 };
                image_buffer.fill([fill_value, fill_value, fill_value, 255]);
            }
            
            // Quit button click
            if quit_button_rect.contains(vec2(mouse_pos.0, mouse_pos.1)) {
                std::process::exit(0);
            }
        }



        // Draw the command summary if help is enabled
        if show_help {
            draw_command_summary(attractors[0].invert);
        }

        // Draw pause indicator
        if paused {
            let text_color = if attractors[0].invert { BLACK } else { WHITE };
            let text = "PAUSED";  // Removed the ⏸ symbol
            let text_size = 16.0;
            let text_dims = measure_text(text, None, text_size as u16, 1.0);
            let padding = 10.0;
            let x = screen_width() - text_dims.width - padding;
            let y = padding + text_dims.height;
            
            // Draw semi-transparent background
            draw_rectangle(
                x - padding/2.0,
                y - text_dims.height - padding/2.0,
                text_dims.width + padding,
                text_dims.height + padding,
                Color::new(0.0, 0.0, 0.0, 0.5)
            );
            
            // Draw text
            draw_text(text, x, y, text_size, text_color);
        }

        // Update key handler for help toggle
        if is_key_pressed(KeyCode::Slash) {
            show_help = !show_help;
            println!("Help display toggled: {}", if show_help { "shown" } else { "hidden" });
        }

        frame_count += 1;
        if frame_count % 60 == 0 && !paused {
            let current_time = get_time();
            let elapsed = current_time - last_fps_time;
            fps = 60.0 / elapsed;
            last_fps_time = current_time;

            for attractor in attractors.iter() {
                attractor.print_stats();
            }
            println!("Performance metrics:");
            println!("  FPS: {:.1}", fps);
            println!("  Frame time: {:.2}ms", elapsed * 1000.0 / 60.0);
            println!("  Iteration time: {:.2}ms ({:.1}%)", 
                iteration_time * 1000.0,
                iteration_time / elapsed * 100.0);
            println!("  Pixel update time: {:.2}ms ({:.1}%)", 
                pixel_update_time * 1000.0,
                pixel_update_time / elapsed * 100.0);
            println!("  Iterations per frame: {}", iterations);
            for (i, attractor) in attractors.iter().enumerate() {
                match attractor.status {
                    AttractorStatus::Running => {
                        println!("Attractor {} Running (frame {})", i, frame_count);
                    }
                    AttractorStatus::Displaying => {
                        let remaining = DISPLAY_DURATION - (get_time() - attractor.display_start_time);
                        println!("Attractor {} Displaying: {:.1} seconds remaining", i, remaining);
                    }
                    AttractorStatus::Fading => {
                        let remaining = FADE_DURATION - (get_time() - attractor.fade_start_time);
                        println!("Attractor {} Fading: {:.1} seconds remaining", i, remaining);
                    }
                }
            }
        }

        next_frame().await
    }
}