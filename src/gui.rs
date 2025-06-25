use eframe::{App, CreationContext, Frame, egui};
use egui_plot::{Legend, Line, Plot, PlotPoints, PlotUi};
use std::time::{Duration, Instant};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System};

pub struct BenchmarkData {
    pub bits: Vec<u32>,
    pub native_gcd_ns: Vec<u128>,
    pub hybrid_gcd_ns: Vec<u128>,
    pub pure_gcd_ns: Vec<u128>,
    pub native_lcm_ns: Vec<u128>,
    pub hybrid_lcm_ns: Vec<u128>,
    pub pure_lcm_ns: Vec<u128>,
    pub lut_mem_kb: u64,
    pub power_table_mem_kb: u64,
    pub primes_mem_kb: u64,

    // neu:
    pub total_iterations: u64,
    pub pure_gcd_hits: u64,
    pub pure_lcm_hits: u64,
}

pub struct BenchmarkApp {
    data: BenchmarkData,
    sys: System,
    last_mem_check: Instant,
    current_mem_kb: u64,
}

impl BenchmarkApp {
    pub fn new(data: BenchmarkData) -> Self {
        let refresh_kind = RefreshKind::nothing().with_processes(ProcessRefreshKind::everything());
        let mut sys = System::new_with_specifics(refresh_kind);
        sys.refresh_processes(ProcessesToUpdate::All, false);
        let pid = Pid::from_u32(std::process::id());
        let current_mem_kb = sys.process(pid).map_or(0, |p| p.memory());

        Self {
            data,
            sys,
            last_mem_check: Instant::now(),
            current_mem_kb,
        }
    }

    fn update_memory(&mut self) {
        if self.last_mem_check.elapsed() >= Duration::from_secs(1) {
            self.sys.refresh_processes(ProcessesToUpdate::All, false);
            let pid = Pid::from_u32(std::process::id());
            self.current_mem_kb = self.sys.process(pid).map_or(0, |p| p.memory());
            self.last_mem_check = Instant::now();
        }
    }
}

impl App for BenchmarkApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        self.update_memory();

        // Memory Window (unchanged)
        egui::Window::new("Memory Usage Breakdown")
            .default_size([300.0, 180.0])
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(format!("Total memory: {} KB", self.current_mem_kb));
                ui.separator();
                ui.label(format!("Lookup Table: {} KB", self.data.lut_mem_kb));
                ui.label(format!("Power Table: {} KB", self.data.power_table_mem_kb));
                ui.label(format!("Primes List: {} KB", self.data.primes_mem_kb));
                let other_mem = self
                    .current_mem_kb
                    .saturating_sub(self.data.lut_mem_kb)
                    .saturating_sub(self.data.power_table_mem_kb)
                    .saturating_sub(self.data.primes_mem_kb);
                ui.label(format!("Other/Process: {} KB", other_mem));
                ui.separator();
                ui.label("Note: Algorithm memory is negligible (stack only)");
            });

        // Neue kleine Info Ecke
        egui::Window::new("Algorithm Stats")
            .default_pos([10.0, 200.0])
            .default_size([240.0, 100.0])
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(format!("Total iterations: {}", self.data.total_iterations));
                ui.label(format!("Pure GCD hits: {}", self.data.pure_gcd_hits));
                ui.label(format!("Pure LCM hits: {}", self.data.pure_lcm_hits));
                let gcd_hit_rate =
                    self.data.pure_gcd_hits as f64 / self.data.total_iterations as f64 * 100.0;
                let lcm_hit_rate =
                    self.data.pure_lcm_hits as f64 / self.data.total_iterations as f64 * 100.0;
                ui.label(format!("GCD hit rate: {:.2} %", gcd_hit_rate));
                ui.label(format!("LCM hit rate: {:.2} %", lcm_hit_rate));
            });

        // GCD Plot (unchanged)
        egui::Window::new("GCD Benchmark Plot")
            .default_size([800.0, 500.0])
            .resizable(true)
            .show(ctx, |ui| {
                Plot::new("GCD Benchmark Plot")
                    .legend(Legend::default())
                    .show(ui, |plot_ui: &mut PlotUi| {
                        let native: PlotPoints = self
                            .data
                            .bits
                            .iter()
                            .zip(self.data.native_gcd_ns.iter())
                            .map(|(&x, &y)| [x as f64, y as f64])
                            .collect();
                        plot_ui.line(Line::new("Native GCD", native));

                        let hybrid: PlotPoints = self
                            .data
                            .bits
                            .iter()
                            .zip(self.data.hybrid_gcd_ns.iter())
                            .map(|(&x, &y)| [x as f64, y as f64])
                            .collect();
                        plot_ui.line(Line::new("Hybrid PFS GCD", hybrid));

                        let pure: PlotPoints = self
                            .data
                            .bits
                            .iter()
                            .zip(self.data.pure_gcd_ns.iter())
                            .map(|(&x, &y)| [x as f64, y as f64])
                            .collect();
                        plot_ui.line(Line::new("Pure PFS GCD", pure));
                    });
            });

        // LCM Plot (unchanged)
        egui::Window::new("LCM Benchmark Plot")
            .default_size([800.0, 500.0])
            .resizable(true)
            .show(ctx, |ui| {
                Plot::new("LCM Benchmark Plot")
                    .legend(Legend::default())
                    .show(ui, |plot_ui: &mut PlotUi| {
                        let native: PlotPoints = self
                            .data
                            .bits
                            .iter()
                            .zip(self.data.native_lcm_ns.iter())
                            .map(|(&x, &y)| [x as f64, y as f64])
                            .collect();
                        plot_ui.line(Line::new("Native LCM", native));

                        let hybrid: PlotPoints = self
                            .data
                            .bits
                            .iter()
                            .zip(self.data.hybrid_lcm_ns.iter())
                            .map(|(&x, &y)| [x as f64, y as f64])
                            .collect();
                        plot_ui.line(Line::new("Hybrid PFS LCM", hybrid));

                        let pure: PlotPoints = self
                            .data
                            .bits
                            .iter()
                            .zip(self.data.pure_lcm_ns.iter())
                            .map(|(&x, &y)| [x as f64, y as f64])
                            .collect();
                        plot_ui.line(Line::new("Pure PFS LCM", pure));
                    });
            });
    }
}

pub fn run_benchmark_gui(data: BenchmarkData) {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Prime Factor Benchmark Visualizer",
        options,
        Box::new(move |_cc: &CreationContext| Ok(Box::new(BenchmarkApp::new(data)))),
    );
}
