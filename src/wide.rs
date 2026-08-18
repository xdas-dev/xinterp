//! Exact comparison of products that can exceed `u128`.
//!
//! The Chebyshev-centre hull trick (`step::chebyshev_pair`) compares intersection points of
//! lines built from tie-point spans: differences of `den` values (up to `u64::MAX` each) times
//! differences of `num` values (up to `i64::MAX` each). The cross-multiplied comparison this
//! needs can reach on the order of `2**129`, past what `i128` or `u128` hold. Rather than pull in
//! a bignum crate for one comparison, multiply into 256 bits by hand -- schoolbook
//! multiplication on 64-bit limbs -- and compare that.

use std::cmp::Ordering;

/// Splits a `u128` into its low and high 64-bit halves.
#[inline]
fn split64(x: u128) -> (u64, u64) {
    (x as u64, (x >> 64) as u64)
}

/// Exact product of two `u128` magnitudes, returned as `(hi, lo)` with
/// `value == hi * 2**128 + lo`.
fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    let (a0, a1) = split64(a);
    let (b0, b1) = split64(b);

    let p00 = a0 as u128 * b0 as u128;
    let p01 = a0 as u128 * b1 as u128;
    let p10 = a1 as u128 * b0 as u128;
    let p11 = a1 as u128 * b1 as u128;

    let r0 = p00 as u64;
    let mid = (p00 >> 64) + (p01 & u64::MAX as u128) + (p10 & u64::MAX as u128);
    let r1 = mid as u64;
    // `mid` sums three values below `2**64`, so its own high part is at most 2 -- nowhere near
    // wide enough to threaten the `hi_sum` below.
    let carry = mid >> 64;
    let hi_sum = p11 + (p01 >> 64) + (p10 >> 64) + carry;
    let r2 = hi_sum as u64;
    let r3 = (hi_sum >> 64) as u64;

    let lo = ((r1 as u128) << 64) | (r0 as u128);
    let hi = ((r3 as u128) << 64) | (r2 as u128);
    (hi, lo)
}

/// Compares `a * b` to `c * d` exactly, for any `i128` operands.
///
/// Splits each product into sign and magnitude, multiplies the magnitudes in 256 bits via
/// [`mul_wide`], and compares. Never overflows, whatever the inputs.
pub fn cross_cmp(a: i128, b: i128, c: i128, d: i128) -> Ordering {
    let sign_ab = a.signum() * b.signum();
    let sign_cd = c.signum() * d.signum();
    if sign_ab != sign_cd {
        return sign_ab.cmp(&sign_cd);
    }
    let mag_ab = mul_wide(a.unsigned_abs(), b.unsigned_abs());
    let mag_cd = mul_wide(c.unsigned_abs(), d.unsigned_abs());
    let magnitude_order = mag_ab.cmp(&mag_cd);
    if sign_ab < 0 {
        magnitude_order.reverse()
    } else {
        magnitude_order
    }
}

/// Compares the fractions `a / d` to `c / b`, exactly, for nonzero `d` and `b`.
pub fn cmp_frac(a: i128, d: i128, c: i128, b: i128) -> Ordering {
    let order = cross_cmp(a, b, c, d);
    if d.signum() * b.signum() < 0 {
        order.reverse()
    } else {
        order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul_wide_matches_small_products() {
        for (a, b) in [(0u128, 0u128), (1, 1), (12345, 6789), (u64::MAX as u128, 2)] {
            let (hi, lo) = mul_wide(a, b);
            assert_eq!(hi, 0);
            assert_eq!(lo, a * b);
        }
    }

    #[test]
    fn test_mul_wide_matches_u128_max_square() {
        // u128::MAX * u128::MAX is exactly representable in 256 bits; check against the
        // known identity n*n = (n-1)*(n+1) + 1 computed with u128 arithmetic on the pieces.
        let n = u128::MAX;
        let (hi, lo) = mul_wide(n, n);
        // n*n = n*(n) ; verify via n*n = n<<0 ... instead reconstruct with a second method:
        // (2^128 - 1)^2 = 2^256 - 2^129 + 1
        // hi*2^128 + lo == 2^256 - 2^129 + 1
        // hi = 2^128 - 2, lo = 1 (since 2^256 - 2*2^128 + 1 = (2^128-2)*2^128 + 1)
        assert_eq!(hi, u128::MAX - 1);
        assert_eq!(lo, 1);
    }

    #[test]
    fn test_cross_cmp_matches_i128_when_no_overflow() {
        let cases: [(i128, i128, i128, i128); 6] = [
            (3, 4, 5, 2),
            (-3, 4, 5, 2),
            (3, -4, -5, 2),
            (0, 5, -3, 2),
            (7, 7, -7, -7),
            (7, 7, 7, 7),
        ];
        for (a, b, c, d) in cases {
            assert_eq!(
                cross_cmp(a, b, c, d),
                (a * b).cmp(&(c * d)),
                "{a} {b} {c} {d}"
            );
        }
    }

    #[test]
    fn test_cross_cmp_beyond_i128() {
        // both products are ~2^129, well past i128::MAX
        let a = (1i128 << 64) - 1;
        let b = (1i128 << 65) - 2;
        let c = a;
        let d = b - 2;
        assert_eq!(cross_cmp(a, b, c, d), Ordering::Greater);
        assert_eq!(cross_cmp(c, d, a, b), Ordering::Less);
        assert_eq!(cross_cmp(a, b, a, b), Ordering::Equal);
    }

    #[test]
    fn test_cmp_frac_matches_f64_when_safe() {
        let cases: [(i128, i128, i128, i128); 5] = [
            (1, 2, 1, 3),
            (-1, 2, 1, 3),
            (1, -2, 1, 3),
            (1, 2, -1, 3),
            (5, 7, 5, 7),
        ];
        for (a, d, c, b) in cases {
            let expected = (a as f64 / d as f64)
                .partial_cmp(&(c as f64 / b as f64))
                .unwrap();
            assert_eq!(cmp_frac(a, d, c, b), expected, "{a} {d} {c} {b}");
        }
    }
}
