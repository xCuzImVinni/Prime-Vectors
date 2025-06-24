#![feature(portable_simd)]

use rand::Rng;
use std::simd::{cmp::SimdOrd, u8x16};
use std::sync::OnceLock;
use std::time::Instant;

mod gui;
use gui::{BenchmarkData, run_benchmark_gui};

const K: usize = 16;
const TRIALS: usize = 1000;
const MAX_LUT: usize = 1 << 18; // 256 KiB Einträge (2 MB bei 17bit, 8 MB bei 19bit)

static PRIMES: OnceLock<Vec<u32>> = OnceLock::new();
static FACTOR_LUT: OnceLock<Box<[([u8; K], u32); MAX_LUT]>> = OnceLock::new();
static POWER_TABLE: OnceLock<[[u64; 64]; K]> = OnceLock::new();

fn init_power_table() -> [[u64; 64]; K] {
    let primes = PRIMES.get().unwrap();
    let mut table = [[1u64; 64]; K];
    for i in 0..K {
        let mut acc = 1u64;
        for e in 1..64 {
            acc = acc.saturating_mul(primes[i] as u64);
            table[i][e] = acc;
        }
    }
    table
}

fn binary_gcd(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let shift = a.trailing_zeros().min(b.trailing_zeros());
    a >>= a.trailing_zeros();
    b >>= b.trailing_zeros();
    while a != b {
        if a > b {
            a -= b;
            a >>= a.trailing_zeros();
        } else {
            b -= a;
            b >>= b.trailing_zeros();
        }
    }
    a << shift
}

fn native_lcm(a: u64, b: u64) -> u64 {
    a / binary_gcd(a, b) * b
}

fn first_k_primes(k: usize) -> Vec<u32> {
    let mut primes = Vec::with_capacity(k);
    let mut candidate = 2;
    while primes.len() < k {
        if primes
            .iter()
            .take_while(|&&p| p * p <= candidate)
            .all(|&p| candidate % p != 0)
        {
            primes.push(candidate);
        }
        candidate += 1;
    }
    primes
}

fn init_lookup_table() -> Box<[([u8; K], u32); MAX_LUT]> {
    let primes = PRIMES.get().expect("PRIMES not initialized");
    let mut table = Box::new([([0u8; K], 1); MAX_LUT]);
    for i in 2..MAX_LUT {
        let mut n = i as u64;
        let mut factor_counts = [0u8; K];
        for (j, &p) in primes.iter().enumerate() {
            let p64 = p as u64;
            while n % p64 == 0 {
                factor_counts[j] += 1;
                n /= p64;
            }
        }
        table[i] = (factor_counts, n as u32);
    }
    table
}

#[inline(always)]
fn factorize(n: u64) -> ([u8; K], u64) {
    if n < MAX_LUT as u64 {
        let (f, r) = FACTOR_LUT.get().unwrap()[n as usize];
        (f, r as u64)
    } else {
        let primes = PRIMES.get().unwrap();
        let mut exponents = [0u8; K];
        let mut rem = n;
        for (i, &p) in primes.iter().enumerate() {
            let p64 = p as u64;
            if p64 * p64 > rem {
                break;
            }
            while rem % p64 == 0 {
                exponents[i] += 1;
                rem /= p64;
            }
        }
        (exponents, rem)
    }
}

#[inline(always)]
fn pfs_to_u64(factors: &[u8; K]) -> u64 {
    let power_table = POWER_TABLE.get().unwrap();
    let mut result = 1u64;
    for i in 0..K {
        result = result.saturating_mul(power_table[i][factors[i] as usize]);
    }
    result
}

fn main() {
    PRIMES.set(first_k_primes(K)).unwrap();
    FACTOR_LUT.get_or_init(init_lookup_table);
    POWER_TABLE.get_or_init(init_power_table);
    let mut rng = rand::thread_rng();

    let mut bits = Vec::new();
    let mut native_gcd_ns = Vec::new();
    let mut pfs_gcd_ns = Vec::new();
    let mut native_lcm_ns = Vec::new();
    let mut pfs_lcm_ns = Vec::new();

    for exp in 4..=32 {
        let max = 1u64 << exp;
        let mut native_gcd_time = 0;
        let mut pfs_gcd_time = 0;
        let mut native_lcm_time = 0;
        let mut pfs_lcm_time = 0;

        for _ in 0..TRIALS {
            let x = rng.gen_range(1..max);
            let y = rng.gen_range(1..max);

            let now = Instant::now();
            let g1 = binary_gcd(x, y);
            native_gcd_time += now.elapsed().as_nanos();

            // --- PFS GCD ---
            let now = Instant::now();
            let (xf, xr) = factorize(x);
            let (yf, yr) = factorize(y);
            let gcd_rem = binary_gcd(xr, yr);

            // SIMD min statt Schleife
            let f_min = u8x16::from_array(xf)
                .simd_min(u8x16::from_array(yf))
                .to_array();
            let g2 = pfs_to_u64(&f_min).saturating_mul(gcd_rem);
            pfs_gcd_time += now.elapsed().as_nanos();

            // --- Native LCM ---
            let now = Instant::now();
            let l1 = native_lcm(x, y);
            native_lcm_time += now.elapsed().as_nanos();

            // --- PFS LCM ---
            let now = Instant::now();
            let lcm_rem = native_lcm(xr, yr); // nicht doppelt aufrufen!
            let f_max = u8x16::from_array(xf)
                .simd_max(u8x16::from_array(yf))
                .to_array();
            let l2 = pfs_to_u64(&f_max).saturating_mul(lcm_rem);
            pfs_lcm_time += now.elapsed().as_nanos();

            debug_assert_eq!(
                g1, g2,
                "GCD mismatch for x={} y={} => {} != {}",
                x, y, g1, g2
            );
            debug_assert_eq!(
                l1, l2,
                "LCM mismatch for x={} y={} => {} != {}",
                x, y, l1, l2
            );
        }

        bits.push(exp);
        native_gcd_ns.push(native_gcd_time / TRIALS as u128);
        pfs_gcd_ns.push(pfs_gcd_time / TRIALS as u128);
        native_lcm_ns.push(native_lcm_time / TRIALS as u128);
        pfs_lcm_ns.push(pfs_lcm_time / TRIALS as u128);

        println!("Processed {} bits", exp);
    }

    let data = BenchmarkData {
        bits,
        native_gcd_ns,
        pfs_gcd_ns,
        native_lcm_ns,
        pfs_lcm_ns,
    };

    let _ = run_benchmark_gui(data);
}
