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
const MAX_LUT: usize = 1 << 16;

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

#[inline(always)]
fn pure_pfs_lcm(xf: [u8; K], yf: [u8; K]) -> u64 {
    // LCM über PFS: lcm = Produkt der Maximalexponenten
    let f_max = u8x16::from_array(xf)
        .simd_max(u8x16::from_array(yf))
        .to_array();
    pfs_to_u64(&f_max)
}

#[inline(always)]
fn hybrid_pfs_lcm(xf: [u8; K], xr: u64, yf: [u8; K], yr: u64) -> u64 {
    // hybrid LCM = pure LCM * lcm von Resten (GCD auf Resten), da restliche Primfaktoren nicht im LUT
    let f_max = u8x16::from_array(xf)
        .simd_max(u8x16::from_array(yf))
        .to_array();
    let base = pfs_to_u64(&f_max);

    if xr == 1 && yr == 1 {
        base
    } else {
        base.saturating_mul(native_lcm(xr, yr))
    }
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
    let primes = PRIMES.get().unwrap();
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
            if p64 > rem || p64 * p64 > rem {
                break;
            }
            while rem % p64 == 0 {
                exponents[i] += 1;
                rem /= p64;
            }
        }
        if rem > 1 {
            None
        } else {
            Some((exponents, rem))
        }
    }
}

#[inline(always)]
fn pfs_to_u64(factors: &[u8; K]) -> u64 {
    let power_table = unsafe { POWER_TABLE.get().unwrap_unchecked() };
    let mut result = 1u64;
    for i in 0..K {
        unsafe {
            let exponent = *factors.get_unchecked(i) as usize;
            let val = *power_table.get_unchecked(i).get_unchecked(exponent);
            result = result.wrapping_mul(val);
        }
    }
    result
}

#[inline(always)]
fn pure_pfs_gcd(xf: [u8; K], yf: [u8; K]) -> u64 {
    let f_min = u8x16::from_array(xf)
        .simd_min(u8x16::from_array(yf))
        .to_array();
    pfs_to_u64(&f_min)
}

#[inline(always)]
fn hybrid_pfs_gcd(xf: [u8; K], xr: u64, yf: [u8; K], yr: u64) -> u64 {
    let f_min = u8x16::from_array(xf)
        .simd_min(u8x16::from_array(yf))
        .to_array();

    let base = pfs_to_u64(&f_min);
    if xr == 1 && yr == 1 {
        base
    } else {
        base.saturating_mul(binary_gcd(xr, yr))
    }
}

fn main() {
    PRIMES.set(first_k_primes(K)).unwrap();
    FACTOR_LUT.get_or_init(init_lookup_table);
    POWER_TABLE.get_or_init(init_power_table);

    let lut_mem = std::mem::size_of_val(&**FACTOR_LUT.get().unwrap()) as u64;
    let power_table_mem = std::mem::size_of_val(POWER_TABLE.get().unwrap()) as u64;
    let primes_mem = PRIMES.get().unwrap().capacity() as u64 * std::mem::size_of::<u32>() as u64;

    let mut bits = Vec::new();
    let mut native_gcd_ns = Vec::new();
    let mut hybrid_gcd_ns = Vec::new();
    let mut pure_gcd_ns = Vec::new();
    let mut native_lcm_ns = Vec::new();
    let mut hybrid_lcm_ns = Vec::new();
    let mut pure_lcm_ns = Vec::new();

    let mut total_iterations = 0u64;
    let mut pure_gcd_hits = 0u64;
    let mut pure_lcm_hits = 0u64;

    for exp in 4..=32 {
        let max = (1 << exp).min(MAX_LUT as u64);

        // Parallele Berechnung mit thread-lokalen Zählern
        let results: Vec<_> = (0..TRIALS)
            .into_par_iter()
            .map_init(
                || StdRng::from_entropy(),
                |rng, _| {
                    let x = rng.gen_range(1..max);
                    let y = rng.gen_range(1..max);

                    let now = Instant::now();
                    let g_native = binary_gcd(x, y);
                    let t_native = now.elapsed().as_nanos();

                    if let (Some((xf, xr)), Some((yf, yr))) = (factorize(x), factorize(y)) {
                        // Hybrid GCD
                        let now = Instant::now();
                        let g_hybrid = hybrid_pfs_gcd(xf, xr, yf, yr);
                        let t_hybrid = now.elapsed().as_nanos();

                        // Pure GCD
                        let now = Instant::now();
                        let g_pure = pure_pfs_gcd(xf, yf);
                        let t_pure = now.elapsed().as_nanos();

                        // Native LCM
                        let now = Instant::now();
                        let l_native = native_lcm(x, y);
                        let t_native_lcm = now.elapsed().as_nanos();

                        // Hybrid LCM
                        let now = Instant::now();
                        let l_hybrid = hybrid_pfs_lcm(xf, xr, yf, yr);
                        let t_hybrid_lcm = now.elapsed().as_nanos();

                        // Pure LCM
                        let now = Instant::now();
                        let l_pure = pure_pfs_lcm(xf, yf);
                        let t_pure_lcm = now.elapsed().as_nanos();

                        // Sanity Checks
                        assert_eq!(
                            l_native, l_hybrid,
                            "Hybrid LCM mismatch for x = {}, y = {}",
                            x, y
                        );
                        assert_eq!(
                            g_native, g_hybrid,
                            "Hybrid GCD mismatch for x = {}, y = {}",
                            x, y
                        );

                        let mut pure_gcd_hit = 0u64;
                        let mut pure_lcm_hit = 0u64;

                        if xr == 1 && yr == 1 {
                            assert_eq!(
                                l_native, l_pure,
                                "Pure LCM mismatch for x = {}, y = {}",
                                x, y
                            );
                            assert_eq!(
                                g_native, g_pure,
                                "Pure GCD mismatch for x = {}, y = {}",
                                x, y
                            );

                            pure_gcd_hit = 1;
                            pure_lcm_hit = 1;
                        }

                        Some((
                            t_native,
                            t_hybrid,
                            t_pure,
                            t_native_lcm,
                            t_hybrid_lcm,
                            t_pure_lcm,
                            pure_gcd_hit,
                            pure_lcm_hit,
                        ))
                    } else {
                        None
                    }
                },
            )
            .filter_map(|x| x)
            .collect();

        // Summieren der Ergebnisse und Treffer
        let mut native_sum = 0u128;
        let mut hybrid_sum = 0u128;
        let mut pure_sum = 0u128;
        let mut native_lcm_sum = 0u128;
        let mut hybrid_lcm_sum = 0u128;
        let mut pure_lcm_sum = 0u128;
        let mut gcd_hits_sum = 0u64;
        let mut lcm_hits_sum = 0u64;

        for (
            t_native,
            t_hybrid,
            t_pure,
            t_native_lcm,
            t_hybrid_lcm,
            t_pure_lcm,
            gcd_hit,
            lcm_hit,
        ) in &results
        {
            native_sum += *t_native;
            hybrid_sum += *t_hybrid;
            pure_sum += *t_pure;

            native_lcm_sum += *t_native_lcm;
            hybrid_lcm_sum += *t_hybrid_lcm;
            pure_lcm_sum += *t_pure_lcm;

            gcd_hits_sum += *gcd_hit;
            lcm_hits_sum += *lcm_hit;
        }

        bits.push(exp);
        native_gcd_ns.push(native_sum / TRIALS as u128);
        hybrid_gcd_ns.push(hybrid_sum / TRIALS as u128);
        pure_gcd_ns.push(pure_sum / TRIALS as u128);

        native_lcm_ns.push(native_lcm_sum / TRIALS as u128);
        hybrid_lcm_ns.push(hybrid_lcm_sum / TRIALS as u128);
        pure_lcm_ns.push(pure_lcm_sum / TRIALS as u128);

        total_iterations += TRIALS as u64;
        pure_gcd_hits += gcd_hits_sum;
        pure_lcm_hits += lcm_hits_sum;
    }

    let data = BenchmarkData {
        bits,
        native_gcd_ns,
        hybrid_gcd_ns,
        pure_gcd_ns,
        native_lcm_ns,
        hybrid_lcm_ns,
        pure_lcm_ns,
        lut_mem_kb: lut_mem / 1024,
        power_table_mem_kb: power_table_mem / 1024,
        primes_mem_kb: primes_mem / 1024,
        total_iterations,
        pure_gcd_hits,
        pure_lcm_hits,
    };

    let _ = run_benchmark_gui(data);
}
