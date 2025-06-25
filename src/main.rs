#![feature(portable_simd)]
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use std::simd::{cmp::SimdOrd, u8x16};
use std::sync::OnceLock;
use std::time::Instant;
use sysinfo::{Pid, Process, System};

mod gui;
use gui::{BenchmarkData, run_benchmark_gui};

const K: usize = 16;
const TRIALS: usize = 1000;
const MAX_LUT: usize = 1 << 20; // 1 Million Einträge (~20 MB Speicher)

static PRIMES: OnceLock<Vec<u32>> = OnceLock::new();
static FACTOR_LUT: OnceLock<Box<[([u8; K], u32)]>> = OnceLock::new();
static POWER_TABLE: OnceLock<[[u64; 64]; K]> = OnceLock::new();

#[inline(always)]
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

#[inline(always)]
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

fn init_lookup_table() -> Box<[([u8; K], u32)]> {
    let primes = PRIMES.get().expect("PRIMES not initialized");

    let mut table = vec![([0u8; K], 1); MAX_LUT];

    table
        .par_iter_mut()
        .enumerate()
        .skip(2)
        .for_each(|(i, entry)| {
            let mut n = i as u64;
            let mut factor_counts = [0u8; K];
            for (j, &p) in primes.iter().enumerate() {
                let p64 = p as u64;
                while n % p64 == 0 {
                    factor_counts[j] += 1;
                    n /= p64;
                }
            }
            *entry = (factor_counts, n as u32);
        });

    table.into_boxed_slice()
}

fn init_power_table() -> [[u64; 64]; K] {
    let primes = PRIMES.get().unwrap();
    let mut table = [[1u64; 64]; K];

    for i in 0..K {
        let mut acc = 1u64;
        let p = primes[i] as u64;
        for e in 1..64 {
            acc = acc.saturating_mul(p);
            table[i][e] = acc;
        }
    }
    table
}

#[inline(always)]
fn factorize(n: u64) -> Option<([u8; K], u64)> {
    if n < MAX_LUT as u64 {
        let (f, r) = unsafe {
            FACTOR_LUT
                .get()
                .unwrap_unchecked()
                .get_unchecked(n as usize)
        };
        Some((*f, *r as u64))
    } else {
        let primes = PRIMES.get().unwrap();
        let mut exponents = [0u8; K];
        let mut rem = n;

        for (i, &p) in primes.iter().enumerate() {
            let p64 = p as u64;
            if p64 > rem {
                break;
            }
            if p64 * p64 > rem {
                break;
            }

            while rem % p64 == 0 {
                exponents[i] += 1;
                rem /= p64;
            }
        }

        if rem > 1 {
            None // nicht vollständig faktorisierbar
        } else {
            Some((exponents, rem))
        }
    }
}

#[inline(always)]
fn pfs_to_u64(factors: &[u8; K]) -> u64 {
    let power_table = unsafe { POWER_TABLE.get().unwrap_unchecked() };
    let mut result = 1u64;

    // Statt saturating_mul, wenn Überlauf ausgeschlossen
    for i in 0..K {
        unsafe {
            let exponent = *factors.get_unchecked(i) as usize;
            let val = *power_table.get_unchecked(i).get_unchecked(exponent);
            result = result.wrapping_mul(val);
        }
    }
    result
}

fn main() {
    PRIMES.set(first_k_primes(K)).unwrap();
    FACTOR_LUT.get_or_init(init_lookup_table);
    POWER_TABLE.get_or_init(init_power_table);

    let mut bits = Vec::new();
    let mut native_gcd_ns = Vec::new();
    let mut pfs_gcd_ns = Vec::new();
    let mut native_lcm_ns = Vec::new();
    let mut pfs_lcm_ns = Vec::new();

    for exp in 4..=32 {
        let max = (1 << exp).min(MAX_LUT as u64);
        let mut native_gcd_time = 0;
        let mut pfs_gcd_time = 0;
        let mut native_lcm_time = 0;
        let mut pfs_lcm_time = 0;

        // Parallele Verarbeitung der Versuche

        let results: Vec<_> = (0..TRIALS)
            .into_par_iter()
            .map_init(
                || StdRng::from_entropy(),
                |rng, _| {
                    let x = rng.gen_range(1..max);
                    let y = rng.gen_range(1..max);

                    // Native GCD
                    let now = Instant::now();
                    let g1 = binary_gcd(x, y);
                    let t_native_gcd = now.elapsed().as_nanos();

                    // --- PFS GCD ---
                    if let (Some((xf, xr)), Some((yf, yr))) = (factorize(x), factorize(y)) {
                        let now = Instant::now();
                        let gcd_rem = binary_gcd(xr, yr);

                        let f_min = u8x16::from_array(xf)
                            .simd_min(u8x16::from_array(yf))
                            .to_array();
                        let g2 = pfs_to_u64(&f_min).saturating_mul(gcd_rem);
                        let t_pfs_gcd = now.elapsed().as_nanos();

                        // --- Native LCM ---
                        let now = Instant::now();
                        let l1 = native_lcm(x, y);
                        let t_native_lcm = now.elapsed().as_nanos();

                        // --- PFS LCM ---
                        let now = Instant::now();
                        let lcm_rem = native_lcm(xr, yr);
                        let f_max = u8x16::from_array(xf)
                            .simd_max(u8x16::from_array(yf))
                            .to_array();
                        let l2 = pfs_to_u64(&f_max).saturating_mul(lcm_rem);
                        let t_pfs_lcm = now.elapsed().as_nanos();

                        assert_eq!(g1, g2, "GCD mismatch: {} vs {} (x={}, y={})", g1, g2, x, y);
                        assert_eq!(l1, l2, "LCM mismatch: {} vs {} (x={}, y={})", l1, l2, x, y);

                        Some((t_native_gcd, t_pfs_gcd, t_native_lcm, t_pfs_lcm))
                    } else {
                        None // Zahlen nicht vollständig faktorisierbar
                    }
                },
            )
            .filter_map(|x| x) // nur gültige Ergebnisse weiterverarbeiten
            .collect();
        // Aggregation der Ergebnisse
        for (t_native_gcd, t_pfs_gcd, t_native_lcm, t_pfs_lcm) in results {
            native_gcd_time += t_native_gcd;
            pfs_gcd_time += t_pfs_gcd;
            native_lcm_time += t_native_lcm;
            pfs_lcm_time += t_pfs_lcm;
        }

        bits.push(exp);
        native_gcd_ns.push(native_gcd_time / TRIALS as u128);
        pfs_gcd_ns.push(pfs_gcd_time / TRIALS as u128);
        native_lcm_ns.push(native_lcm_time / TRIALS as u128);
        pfs_lcm_ns.push(pfs_lcm_time / TRIALS as u128);

        println!(
            "Bits: {:2} | GCD: {:6}ns (PFS) vs {:6}ns (Native) | LCM: {:6}ns (PFS) vs {:6}ns (Native)",
            exp,
            pfs_gcd_time / TRIALS as u128,
            native_gcd_time / TRIALS as u128,
            pfs_lcm_time / TRIALS as u128,
            native_lcm_time / TRIALS as u128,
        );
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
