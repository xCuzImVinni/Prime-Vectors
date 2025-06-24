use eframe::{App, CreationContext, Frame, egui};
use egui_plot::{Legend, Line, Plot, PlotPoints, PlotUi};

pub struct BenchmarkData {
    pub bits: Vec<u32>,
    pub native_gcd_ns: Vec<u128>,
    pub pfs_gcd_ns: Vec<u128>,
    pub native_lcm_ns: Vec<u128>,
    pub pfs_lcm_ns: Vec<u128>,
}

pub struct BenchmarkApp {
    data: BenchmarkData,
}

impl BenchmarkApp {
    pub fn new(data: BenchmarkData) -> Self {
        Self { data }
    }
}

impl App for BenchmarkApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        // GCD Plot Window
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

        // LCM Plot Window
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
        (Box::new(move |_cc: &CreationContext| Ok(Box::new(BenchmarkApp::new(data))))),
    );
}
