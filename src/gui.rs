use eframe::{App, CreationContext, Frame, egui};
use egui_plot::{Legend, Line, Plot, PlotPoints, PlotUi};
use std::time::{Duration, Instant};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System};

pub struct BenchmarkData {
    pub bits: Vec<u32>,
    pub native_gcd_ns: Vec<u128>,
    pub pfs_gcd_ns: Vec<u128>,
    pub native_lcm_ns: Vec<u128>,
    pub pfs_lcm_ns: Vec<u128>,
}

pub struct BenchmarkApp {
    data: BenchmarkData,
    sys: System,
    last_mem_check: Instant,
    current_mem_kb: u64,
}

impl BenchmarkApp {
    pub fn new(data: BenchmarkData) -> Self {
        // Nur Prozessinformationen aktualisieren
        let refresh_kind = RefreshKind::nothing().with_processes(ProcessRefreshKind::everything());
        let mut sys = System::new_with_specifics(refresh_kind);

        // Prozesse einmal initial aktualisieren
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
            // Prozesse aktualisieren, CPU-Refresh auf false
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

        egui::Window::new("Memory Usage")
            .default_size([300.0, 80.0])
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(format!(
                    "Current memory usage: {} KB (~{} MB)",
                    self.current_mem_kb,
                    self.current_mem_kb / 1024
                ));
            });

        egui::Window::new("GCD Plot")
            .default_size([600.0, 400.0])
            .resizable(true)
            .show(ctx, |ui| {
                Plot::new("GCD Benchmark").legend(Legend::default()).show(
                    ui,
                    |plot_ui: &mut PlotUi| {
                        let native: PlotPoints = self
                            .data
                            .bits
                            .iter()
                            .zip(self.data.native_gcd_ns.iter())
                            .map(|(&x, &y)| [x as f64, y as f64])
                            .collect();
                        plot_ui.line(Line::new("Native GCD", native));

                        let pfs: PlotPoints = self
                            .data
                            .bits
                            .iter()
                            .zip(self.data.pfs_gcd_ns.iter())
                            .map(|(&x, &y)| [x as f64, y as f64])
                            .collect();
                        plot_ui.line(Line::new("PFS GCD", pfs));
                    },
                );
            });

        egui::Window::new("LCM Plot")
            .default_size([600.0, 400.0])
            .resizable(true)
            .show(ctx, |ui| {
                Plot::new("LCM Benchmark").legend(Legend::default()).show(
                    ui,
                    |plot_ui: &mut PlotUi| {
                        let native: PlotPoints = self
                            .data
                            .bits
                            .iter()
                            .zip(self.data.native_lcm_ns.iter())
                            .map(|(&x, &y)| [x as f64, y as f64])
                            .collect();
                        plot_ui.line(Line::new("Native LCM", native));

                        let pfs: PlotPoints = self
                            .data
                            .bits
                            .iter()
                            .zip(self.data.pfs_lcm_ns.iter())
                            .map(|(&x, &y)| [x as f64, y as f64])
                            .collect();
                        plot_ui.line(Line::new("PFS LCM", pfs));
                    },
                );
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
