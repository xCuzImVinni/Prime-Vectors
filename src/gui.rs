// Core GUI framework and plotting crates
use eframe::{App, CreationContext, Frame, egui};
use egui_plot::{Legend, Line, Plot, PlotPoint, PlotPoints, PlotUi};
use std::time::{Duration, Instant};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System};

/// Enum to switch between GCD and LCM metrics
#[derive(Debug, Clone, Copy, PartialEq)]
enum Metric {
    GCD,
    LCM,
}

/// Enum to distinguish algorithmic styles
#[derive(Debug, Clone, Copy, PartialEq)]
enum Style {
    Native, // pure Rust implementation
    Hybrid, // mix of native and lookup
    Pure,   // full lookup table approach
}

/// Container for all benchmark-related data and metrics
pub struct BenchmarkData {
    pub bits: Vec<u32>,           // Input sizes in bits (e.g. 8, 16, 32)
    pub native_gcd_ns: Vec<u128>, // Execution times for native GCD
    pub hybrid_gcd_ns: Vec<u128>, // Execution times for hybrid GCD
    pub pure_gcd_ns: Vec<u128>,   // Execution times for pure GCD
    pub native_lcm_ns: Vec<u128>, // Execution times for native LCM
    pub hybrid_lcm_ns: Vec<u128>, // Execution times for hybrid LCM
    pub pure_lcm_ns: Vec<u128>,   // Execution times for pure LCM
    pub lut_mem_kb: u64,          // Lookup table memory in KB
    pub power_table_mem_kb: u64,  // Power table memory in KB
    pub primes_mem_kb: u64,       // Prime list memory in KB
    pub total_iterations: u64,    // Total test iterations
    pub pure_gcd_hits: u64,       // Cache hits for pure GCD
    pub pure_lcm_hits: u64,       // Cache hits for pure LCM
}

/// Main application state that drives the GUI
pub struct BenchmarkApp {
    data: BenchmarkData,     // All collected data
    sys: System,             // System memory monitoring
    last_mem_check: Instant, // Timer for throttling memory polling
    current_mem_kb: u64,     // Live memory usage in KB

    // Toggles for visible plot lines
    show_native: bool,
    show_hybrid: bool,
    show_pure: bool,

    selected_metric: Metric, // Whether we are viewing GCD or LCM
    show_info: bool,         // Show algorithm information panel
    show_stats: bool,        // Show numeric statistics like hit rates
}

impl BenchmarkApp {
    /// Initialize the GUI state and system resource tracking
    pub fn new(data: BenchmarkData) -> Self {
        // Configure sysinfo to only refresh process data
        let refresh_kind = RefreshKind::nothing().with_processes(ProcessRefreshKind::everything());
        let mut sys = System::new_with_specifics(refresh_kind);
        sys.refresh_processes(ProcessesToUpdate::All, false);

        // Fetch current memory usage of this process
        let pid = Pid::from_u32(std::process::id());
        let current_mem_kb = sys.process(pid).map_or(0, |p| p.memory());

        Self {
            data,
            sys,
            last_mem_check: Instant::now(),
            current_mem_kb,
            show_native: true,
            show_hybrid: true,
            show_pure: true,
            selected_metric: Metric::GCD,
            show_info: false,
            show_stats: true,
        }
    }

    /// Refresh memory usage once per second to reduce overhead
    fn update_memory(&mut self) {
        if self.last_mem_check.elapsed() >= Duration::from_secs(1) {
            self.sys.refresh_processes(ProcessesToUpdate::All, false);
            let pid = Pid::from_u32(std::process::id());
            self.current_mem_kb = self.sys.process(pid).map_or(0, |p| p.memory());
            self.last_mem_check = Instant::now();
        }
    }

    /// Retrieve y-values for the plot based on current style and metric
    fn get_plot_data(&self, style: Style) -> Vec<f64> {
        match (self.selected_metric, style) {
            (Metric::GCD, Style::Native) => {
                self.data.native_gcd_ns.iter().map(|&x| x as f64).collect()
            }
            (Metric::GCD, Style::Hybrid) => {
                self.data.hybrid_gcd_ns.iter().map(|&x| x as f64).collect()
            }
            (Metric::GCD, Style::Pure) => self.data.pure_gcd_ns.iter().map(|&x| x as f64).collect(),
            (Metric::LCM, Style::Native) => {
                self.data.native_lcm_ns.iter().map(|&x| x as f64).collect()
            }
            (Metric::LCM, Style::Hybrid) => {
                self.data.hybrid_lcm_ns.iter().map(|&x| x as f64).collect()
            }
            (Metric::LCM, Style::Pure) => self.data.pure_lcm_ns.iter().map(|&x| x as f64).collect(),
        }
    }
}

impl App for BenchmarkApp {
    /// This method is called every frame to update UI and state
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        self.update_memory(); // Update memory if needed

        // Left side panel with UI controls
        egui::SidePanel::left("controls").show(ctx, |ui| {
            use egui::RichText;
            ui.heading(RichText::new("Settings").size(24.0));

            egui::ComboBox::from_id_source("metric_select")
                .selected_text(format!("{:?}", self.selected_metric))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.selected_metric, Metric::GCD, "GCD");
                    ui.selectable_value(&mut self.selected_metric, Metric::LCM, "LCM");
                });

            ui.separator();
            ui.label(RichText::new("Toggle styles:").size(20.0));
            ui.checkbox(&mut self.show_native, "Native");
            ui.checkbox(&mut self.show_hybrid, "Hybrid");
            ui.checkbox(&mut self.show_pure, "Pure");

            ui.separator();
            ui.checkbox(&mut self.show_info, "Show Info View");
            ui.checkbox(&mut self.show_stats, "Show Stats");

            if self.show_stats && !self.show_info {
                ui.separator();
                ui.label(
                    RichText::new(format!("Total iterations: {}", self.data.total_iterations))
                        .size(18.0),
                );
                ui.label(
                    RichText::new(format!("Pure GCD hits: {}", self.data.pure_gcd_hits)).size(18.0),
                );
                ui.label(
                    RichText::new(format!("Pure LCM hits: {}", self.data.pure_lcm_hits)).size(18.0),
                );

                let gcd_hit_rate =
                    self.data.pure_gcd_hits as f64 / self.data.total_iterations as f64 * 100.0;
                let lcm_hit_rate =
                    self.data.pure_lcm_hits as f64 / self.data.total_iterations as f64 * 100.0;

                ui.label(RichText::new(format!("GCD hit rate: {:.2}%", gcd_hit_rate)).size(18.0));
                ui.label(RichText::new(format!("LCM hit rate: {:.2}%", lcm_hit_rate)).size(18.0));

                ui.separator();
                ui.label(RichText::new(format!("Memory: {} KB", self.current_mem_kb)).size(18.0));
                ui.label(RichText::new(format!("LUT: {} KB", self.data.lut_mem_kb)).size(18.0));
                ui.label(
                    RichText::new(format!("Powers: {} KB", self.data.power_table_mem_kb))
                        .size(18.0),
                );
                ui.label(
                    RichText::new(format!("Primes: {} KB", self.data.primes_mem_kb)).size(18.0),
                );
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            use egui::RichText;
            if self.show_info {
                ui.heading(RichText::new("Algorithm Details").size(24.0));
                ui.label(
                    RichText::new(
                        "Native: Optimized standard implementation without lookup tables.",
                    )
                    .size(18.0),
                );
                ui.label(
                    RichText::new("Hybrid: Mixed implementation using native + lookup.").size(18.0),
                );
                ui.label(RichText::new("Pure: Fully accelerated via PFS and SIMD.").size(18.0));
                ui.separator();
                ui.label(RichText::new("Used Resources:").size(20.0));
                ui.label(
                    RichText::new("- LUT: Prime factor lookup table (e.g., 2-byte vectors).")
                        .size(16.0),
                );
                ui.label(RichText::new("- PowerTable: Precomputed prime powers.").size(16.0));
                ui.label(
                    RichText::new("- Primes: List of first n primes for decomposition.").size(16.0),
                );
            } else {
                Plot::new("benchmark_plot")
                    .legend(Legend::default())
                    .include_y(0.0)
                    .include_y(200.0)
                    .include_x(0.0)
                    .include_x(32.0)
                    .show_x(true)
                    .show_y(true)
                    .show(ui, |plot_ui: &mut PlotUi| {
                        for (style, enabled) in [
                            (Style::Native, self.show_native),
                            (Style::Hybrid, self.show_hybrid),
                            (Style::Pure, self.show_pure),
                        ] {
                            if !enabled {
                                continue;
                            }

                            let points: PlotPoints = self
                                .data
                                .bits
                                .iter()
                                .cloned()
                                .zip(self.get_plot_data(style))
                                .map(|(x, y)| [x as f64, y])
                                .collect();

                            let label = format!("{:?} {:?}", style, self.selected_metric);
                            plot_ui.line(Line::new(label, points));
                        }
                    });
            }
        });
    }
}

/// Launch the GUI application with given benchmark data
pub fn run_benchmark_gui(data: BenchmarkData) {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Prime Factor Benchmark Visualizer",
        options,
        Box::new(move |_cc: &CreationContext| Ok(Box::new(BenchmarkApp::new(data)))),
    );
}
