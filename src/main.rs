use rand::Rng;
use std::arch::x86_64::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::{Duration, Instant};

mod gui;
use gui::{run_benchmark_gui, BenchmarkData};

const K: usize = 16; // first 16 primes
const TRIALS: usize = 1000;

fn first_k_primes(k: usize) -> Vec<u32> {
    let mut primes = Vec::new();
    let mut num = 2;
    while primes.len() < k {
        let mut is_prime = true;
        for &p in &primes {
            if (p as usize).pow(2) > num {
                break;
            }
            if num % p as usize == 0 {
                is_prime = false;
                break;
            }
        }
        if is_prime {
            primes.push(num as u32);
        }
        num += 1;
    }
    primes
}

fn factor_to_array(mut n: u32, primes: &[u32], exps: &mut [u8; K]) -> u32 {
    *exps = [0u8; K];
    for (i, &p) in primes.iter().enumerate() {
        while n % p == 0 {
            exps[i] = exps[i].wrapping_add(1);
            n /= p;
        }
    }
    n
}

fn reconstruct_array(exps: &[u8; K], primes: &[u32], leftover: u32) -> u64 {
    let mut res = leftover as u64;
    for i in 0..K {
        let e = exps[i] as usize;
        if e > 0 {
            res *= (0..e).fold(1u64, |a, _| a * primes[i] as u64);
        }
    }
    res
}

unsafe fn simd_min(a: *const u8, b: *const u8, out: *mut u8) {
    let va = _mm_loadu_si128(a as *const __m128i);
    let vb = _mm_loadu_si128(b as *const __m128i);
    let vc = _mm_min_epu8(va, vb);
    _mm_storeu_si128(out as *mut __m128i, vc);
}
unsafe fn simd_add(a: *const u8, b: *const u8, out: *mut u8) {
    let va = _mm_loadu_si128(a as *const __m128i);
    let vb = _mm_loadu_si128(b as *const __m128i);
    let vc = _mm_add_epi8(va, vb);
    _mm_storeu_si128(out as *mut __m128i, vc);
}
unsafe fn simd_sub(a: *const u8, b: *const u8, out: *mut u8) {
    let va = _mm_loadu_si128(a as *const __m128i);
    let vb = _mm_loadu_si128(b as *const __m128i);
    let vc = _mm_sub_epi8(va, vb);
    _mm_storeu_si128(out as *mut __m128i, vc);
}

fn gcd_euclid(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn load_benchmark_data(filename: &str) -> Result<BenchmarkData, Box<dyn std::error::Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let mut bits = Vec::new();
    let mut native_gcd_ns = Vec::new();
    let mut pfs_gcd_ns = Vec::new();
    let mut native_lcm_ns = Vec::new();
    let mut pfs_lcm_ns = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 {
            continue; // Header überspringen
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() == 5 {
            bits.push(cols[0].parse()?);
            native_gcd_ns.push(cols[1].parse()?);
            pfs_gcd_ns.push(cols[2].parse()?);
            native_lcm_ns.push(cols[3].parse()?);
            pfs_lcm_ns.push(cols[4].parse()?);
        }
    }

    Ok(BenchmarkData {
        bits,
        native_gcd_ns,
        pfs_gcd_ns,
        native_lcm_ns,
        pfs_lcm_ns,
    })
}

fn main() -> Result<(), eframe::Error> {
    let primes = first_k_primes(K);
    let mut rng = rand::thread_rng();
    let mut exp_a = [0u8; K];
    let mut exp_b = [0u8; K];
    let mut exp_g = [0u8; K];
    let mut exp_ab = [0u8; K];
    let mut exp_lcm = [0u8; K];

    let mut file = File::create("benchmark_output.csv").expect("Unable to create file");
    writeln!(
        file,
        "bits,native_gcd_ns,pfs_gcd_ns,native_lcm_ns,pfs_lcm_ns"
    )
    .unwrap();

    // Vektoren zum Speichern der Ergebnisse für die GUI
    let mut bits_vec = Vec::new();
    let mut native_gcd_vec = Vec::new();
    let mut pfs_gcd_vec = Vec::new();
    let mut native_lcm_vec = Vec::new();
    let mut pfs_lcm_vec = Vec::new();

    for bits in 16..=64 {
        let mut native_gcd = Duration::ZERO;
        let mut pfs_gcd = Duration::ZERO;
        let mut native_lcm_total = Duration::ZERO;
        let mut pfs_lcm_total = Duration::ZERO;

        let mask: u64 = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };

        for _ in 0..TRIALS {
            let raw_a = rng.gen::<u64>() & mask;
            let raw_b = rng.gen::<u64>() & mask;
            let a = raw_a.min(u32::MAX as u64) as u32;
            let b = raw_b.min(u32::MAX as u64) as u32;

            if a == 0 || b == 0 {
                continue;
            }

            let start = Instant::now();
            let g = gcd_euclid(a, b);
            native_gcd += start.elapsed();

            let start = Instant::now();
            let _ = (a as u64 / g as u64) * b as u64;
            native_lcm_total += start.elapsed();

            let _ = factor_to_array(a, &primes, &mut exp_a);
            let _ = factor_to_array(b, &primes, &mut exp_b);

            let start = Instant::now();
            unsafe {
                simd_min(exp_a.as_ptr(), exp_b.as_ptr(), exp_g.as_mut_ptr());
            }
            pfs_gcd += start.elapsed();

            unsafe {
                simd_add(exp_a.as_ptr(), exp_b.as_ptr(), exp_ab.as_mut_ptr());
                simd_sub(exp_ab.as_ptr(), exp_g.as_ptr(), exp_lcm.as_mut_ptr());
            }
            let start = Instant::now();
            let _ = reconstruct_array(&exp_lcm, &primes, 1);
            pfs_lcm_total += start.elapsed();
        }

        let avg_native_gcd = native_gcd.as_nanos() / TRIALS as u128;
        let avg_pfs_gcd = pfs_gcd.as_nanos() / TRIALS as u128;
        let avg_native_lcm = native_lcm_total.as_nanos() / TRIALS as u128;
        let avg_pfs_lcm = pfs_lcm_total.as_nanos() / TRIALS as u128;

        // Daten für CSV speichern
        writeln!(
            file,
            "{},{},{},{},{}",
            bits, avg_native_gcd, avg_pfs_gcd, avg_native_lcm, avg_pfs_lcm
        )
        .unwrap();

        // Daten für GUI sammeln
        bits_vec.push(bits as u32);
        native_gcd_vec.push(avg_native_gcd);
        pfs_gcd_vec.push(avg_pfs_gcd);
        native_lcm_vec.push(avg_native_lcm);
        pfs_lcm_vec.push(avg_pfs_lcm);

        println!(
            "Bits: {}\tGCD native: {} ns\tPFS: {} ns\tLCM native: {} ns\tPFS: {} ns",
            bits, avg_native_gcd, avg_pfs_gcd, avg_native_lcm, avg_pfs_lcm
        );
    }

    println!("Done. Output written to benchmark_output.csv");

    // Daten für GUI aufbereiten
    let data = BenchmarkData {
        bits: bits_vec,
        native_gcd_ns: native_gcd_vec,
        pfs_gcd_ns: pfs_gcd_vec,
        native_lcm_ns: native_lcm_vec,
        pfs_lcm_ns: pfs_lcm_vec,
    };

    run_benchmark_gui(data)
}
