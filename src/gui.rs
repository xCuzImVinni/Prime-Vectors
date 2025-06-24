use eframe::{App, Frame, egui};
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
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("GCD/LCM Benchmark Ergebnisse");
            ui.separator();

            self.plot(ui, "GCD Vergleich", |app, plot_ui| {
                app.add_gcd_lines(plot_ui);
            });

            self.plot(ui, "LCM Vergleich", |app, plot_ui| {
                app.add_lcm_lines(plot_ui);
            });

            ui.collapsing("Algorithmische Details", |ui| {
                ui.label("GCD Native: O(log(min(a,b)))");
                ui.label("GCD PFS: O(k) (k = Anzahl Primfaktoren)");
                ui.label("LCM Native: O(1) nach GCD-Berechnung");
                ui.label("LCM PFS: O(k)");
                ui.label("Speicherbedarf PFS: 16 Bytes (für K=16 Primfaktoren)");
            });
        });
    }
}

impl BenchmarkApp {
    fn plot<F: FnMut(&mut Self, &mut PlotUi)>(
        &mut self,
        ui: &mut egui::Ui,
        title: &str,
        mut add_lines: F,
    ) {
        Plot::new(title)
            .legend(Legend::default())
            .allow_zoom(true)
            .allow_drag(true)
            .allow_scroll(false)
            .show_axes([true, true])
            .auto_bounds(true)
            .show(ui, |plot_ui| {
                add_lines(self, plot_ui);
            });
    }

    fn add_gcd_lines(&self, plot_ui: &mut PlotUi) {
        let native_points: Vec<[f64; 2]> = self
            .data
            .bits
            .iter()
            .zip(self.data.native_gcd_ns.iter())
            .map(|(&x, &y)| [x as f64, y as f64])
            .collect();

        let pfs_points: Vec<[f64; 2]> = self
            .data
            .bits
            .iter()
            .zip(self.data.pfs_gcd_ns.iter())
            .map(|(&x, &y)| [x as f64, y as f64])
            .collect();

        // Korrigiert: Zwei Argumente für Line::new()
        let native_line = Line::new(
            "Native GCD",                    // Name als erstes Argument
            PlotPoints::from(native_points), // Daten als zweites Argument
        )
        .color(egui::Color32::RED);

        let pfs_line = Line::new(
            "PFS GCD",                    // Name als erstes Argument
            PlotPoints::from(pfs_points), // Daten als zweites Argument
        )
        .color(egui::Color32::GREEN);

        plot_ui.line(native_line);
        plot_ui.line(pfs_line);
    }

    fn add_lcm_lines(&self, plot_ui: &mut PlotUi) {
        let native_points: Vec<[f64; 2]> = self
            .data
            .bits
            .iter()
            .zip(self.data.native_lcm_ns.iter())
            .map(|(&x, &y)| [x as f64, y as f64])
            .collect();

        let pfs_points: Vec<[f64; 2]> = self
            .data
            .bits
            .iter()
            .zip(self.data.pfs_lcm_ns.iter())
            .map(|(&x, &y)| [x as f64, y as f64])
            .collect();

        // Korrigiert: Zwei Argumente für Line::new()
        let native_line = Line::new(
            "Native LCM",                    // Name als erstes Argument
            PlotPoints::from(native_points), // Daten als zweites Argument
        )
        .color(egui::Color32::BLUE);

        let pfs_line = Line::new(
            "PFS LCM",                    // Name als erstes Argument
            PlotPoints::from(pfs_points), // Daten als zweites Argument
        )
        .color(egui::Color32::YELLOW);

        plot_ui.line(native_line);
        plot_ui.line(pfs_line);
    }
}

pub fn run_benchmark_gui(data: BenchmarkData) -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "GCD/LCM Benchmark Analyse",
        options,
        Box::new(move |_cc| Ok(Box::new(BenchmarkApp::new(data)))),
    )
}
