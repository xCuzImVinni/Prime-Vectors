//PFS := Prime Factorization System, native := the usual way
use rand::Rng;
use std::arch::x86_64::*;
use std::fs::File;
use std::io::Write;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

mod gui;
use gui::{BenchmarkData, run_benchmark_gui};

const K: usize = 20; // Erhöht auf 20 Primzahlen
const TRIALS: usize = 1000;
const MAX_LUT: usize = 1 << 16; // 65536

static FACTOR_LUT: OnceLock<Vec<([u8; K], u32)>> = OnceLock::new();

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

fn init_lookup_table(primes: &[u32]) {
    let mut table = vec![([0u8; K], 0); MAX_LUT];
    for n in 1..MAX_LUT {
        let mut leftover = n as u32;
        let mut exps = [0u8; K];

        for (i, &p) in primes.iter().enumerate() {
            while leftover % p == 0 {
                exps[i] = exps[i].saturating_add(1);
                leftover /= p;
            }
        }
        table[n] = (exps, leftover);
    }
    FACTOR_LUT.set(table).unwrap();
}

fn factor_to_array(n: u32, primes: &[u32], exps: &mut [u8; K]) -> u32 {
    if n < MAX_LUT as u32 {
        let table = FACTOR_LUT.get().expect("LUT not initialized");
        let (lookup_exp, leftover) = table[n as usize];
        *exps = lookup_exp;
        leftover
    } else {
        exps.iter_mut().for_each(|e| *e = 0);
        let mut leftover = n;

        // Faktorisierung für Primzahlen <= sqrt(n)
        for (i, &p) in primes.iter().enumerate() {
            if p * p > leftover {
                break;
            }
            while leftover % p == 0 {
                exps[i] = exps[i].saturating_add(1);
                leftover /= p;
            }
        }

        // Nachbearbeitung für große Primfaktoren
        if leftover > 1 {
            if let Some(pos) = primes.iter().position(|&prime| prime == leftover) {
                exps[pos] = exps[pos].saturating_add(1);
                leftover = 1;
            }
        }

        leftover
    }
}

fn reconstruct_array(exps: &[u8; K], primes: &[u32], leftover: u32) -> u64 {
    let mut res = leftover as u64;
    for i in 0..K {
        if exps[i] > 0 {
            let p = primes[i] as u64;
            res *= p.pow(exps[i] as u32);
        }
    }
    res
}

fn simd_min(a: *const u8, b: *const u8, out: *mut u8) {
    unsafe {
        let va = _mm_loadu_si128(a as *const __m128i);
        let vb = _mm_loadu_si128(b as *const __m128i);
        let vc = _mm_min_epu8(va, vb);
        _mm_storeu_si128(out as *mut __m128i, vc);
    }
}

fn simd_add(a: *const u8, b: *const u8, out: *mut u8) {
    unsafe {
        let va = _mm_loadu_si128(a as *const __m128i);
        let vb = _mm_loadu_si128(b as *const __m128i);
        let vc = _mm_add_epi8(va, vb);
        _mm_storeu_si128(out as *mut __m128i, vc);
    }
}

fn simd_sub(a: *const u8, b: *const u8, out: *mut u8) {
    unsafe {
        let va = _mm_loadu_si128(a as *const __m128i);
        let vb = _mm_loadu_si128(b as *const __m128i);
        let vc = _mm_sub_epi8(va, vb);
        _mm_storeu_si128(out as *mut __m128i, vc);
    }
}

fn gcd_euclid(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn main() -> Result<(), eframe::Error> {
    let primes = first_k_primes(K);
    init_lookup_table(&primes);
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
            let raw_a: u64 = rng.r#gen::<u64>() & mask;
            let raw_b: u64 = rng.r#gen::<u64>() & mask;
            let a = raw_a.min(u32::MAX as u64) as u32;
            let b = raw_b.min(u32::MAX as u64) as u32;

            if a == 0 || b == 0 {
                continue;
            }

            // Native GCD & LCM
            let start = Instant::now();
            let g = gcd_euclid(a, b);
            native_gcd += start.elapsed();

            let start = Instant::now();
            let _lcm = (a as u64 * b as u64) / g as u64;
            native_lcm_total += start.elapsed();

            // PFS GCD & LCM
            let start = Instant::now();
            let leftover_a = factor_to_array(a, &primes, &mut exp_a);
            let leftover_b = factor_to_array(b, &primes, &mut exp_b);
            simd_min(exp_a.as_ptr(), exp_b.as_ptr(), exp_g.as_mut_ptr());
            let leftover_gcd = gcd_euclid(leftover_a, leftover_b);
            let g2 = reconstruct_array(&exp_g, &primes, leftover_gcd);
            pfs_gcd += start.elapsed();

            // Korrektheitsprüfung
            if u64::from(g) != g2 {
                eprintln!(
                    "GCD mismatch: {} vs {}\na={} b={}\nexp_a={:?}\nexp_b={:?}\nleftover_a={} leftover_b={}\nprimes={:?}",
                    g, g2, a, b, exp_a, exp_b, leftover_a, leftover_b, primes
                );
            }

            simd_add(exp_a.as_ptr(), exp_b.as_ptr(), exp_ab.as_mut_ptr());
            simd_sub(exp_ab.as_ptr(), exp_g.as_ptr(), exp_lcm.as_mut_ptr());

            let start = Instant::now();
            let _ = reconstruct_array(&exp_lcm, &primes, 1);
            pfs_lcm_total += start.elapsed();
        }

        let avg_native_gcd = native_gcd.as_nanos() / TRIALS as u128;
        let avg_pfs_gcd = pfs_gcd.as_nanos() / TRIALS as u128;
        let avg_native_lcm = native_lcm_total.as_nanos() / TRIALS as u128;
        let avg_pfs_lcm = pfs_lcm_total.as_nanos() / TRIALS as u128;

        writeln!(
            file,
            "{},{},{},{},{}",
            bits, avg_native_gcd, avg_pfs_gcd, avg_native_lcm, avg_pfs_lcm
        )
        .unwrap();

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

    let data = BenchmarkData {
        bits: bits_vec,
        native_gcd_ns: native_gcd_vec,
        pfs_gcd_ns: pfs_gcd_vec,
        native_lcm_ns: native_lcm_vec,
        pfs_lcm_ns: pfs_lcm_vec,
    };

    run_benchmark_gui(data)
}
