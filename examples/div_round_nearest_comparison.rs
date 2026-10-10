use ezkl::circuit::ops::layouts::round_half_away_from_zero_divmod;
use ezkl::fieldutils::{felt_to_integer_rep, integer_rep_to_felt, IntegerRep};
use ezkl::tensor::ops::nonlinearities::const_div;
use ezkl::tensor::Tensor;
use halo2curves::bn256::Fr as F;

fn q_old(input: IntegerRep, divisor: IntegerRep) -> IntegerRep {
    let input_felt: F = integer_rep_to_felt(input);
    let divisor_felt: F = integer_rep_to_felt(divisor);
    let input_evals = Tensor::<IntegerRep>::new(Some(&[felt_to_integer_rep(input_felt)]), &[1])
        .expect("tensor construction");
    let denom_as_f64 = felt_to_integer_rep(divisor_felt) as f64;
    const_div(&input_evals, denom_as_f64)[0]
}

fn q_new(input: IntegerRep, divisor: IntegerRep) -> Option<IntegerRep> {
    if input > IntegerRep::MAX - divisor || input < IntegerRep::MIN + divisor {
        return None;
    }
    let input_t = Tensor::<IntegerRep>::new(Some(&[input]), &[1]).unwrap();
    let (q, _r) = round_half_away_from_zero_divmod(&input_t, divisor);
    Some(q[0])
}

struct Case {
    label: &'static str,
    divisor: IntegerRep,
    input: IntegerRep,
}

fn cases_for_divisor(d: IntegerRep) -> Vec<Case> {
    let half = d / 2;
    let quarter = d / 4;
    vec![
        Case {
            label: "zero",
            divisor: d,
            input: 0,
        },
        Case {
            label: "pos_multiple_5x",
            divisor: d,
            input: 5 * d,
        },
        Case {
            label: "neg_multiple_5x",
            divisor: d,
            input: -5 * d,
        },
        Case {
            label: "pos_one",
            divisor: d,
            input: 1,
        },
        Case {
            label: "neg_one",
            divisor: d,
            input: -1,
        },
        Case {
            label: "pos_div_minus_1",
            divisor: d,
            input: d - 1,
        },
        Case {
            label: "neg_div_minus_1",
            divisor: d,
            input: -(d - 1),
        },
        Case {
            label: "pos_div",
            divisor: d,
            input: d,
        },
        Case {
            label: "neg_div",
            divisor: d,
            input: -d,
        },
        Case {
            label: "pos_div_plus_1",
            divisor: d,
            input: d + 1,
        },
        Case {
            label: "neg_div_plus_1",
            divisor: d,
            input: -(d + 1),
        },
        Case {
            label: "pos_normal_frac",
            divisor: d,
            input: 3 * d + d / 3,
        },
        Case {
            label: "neg_normal_frac",
            divisor: d,
            input: -(3 * d + d / 3),
        },
        Case {
            label: "pos_ratio_0.25",
            divisor: d,
            input: quarter,
        },
        Case {
            label: "neg_ratio_0.25",
            divisor: d,
            input: -quarter,
        },
        Case {
            label: "pos_ratio_0.5_TIE",
            divisor: d,
            input: half,
        },
        Case {
            label: "neg_ratio_0.5_TIE",
            divisor: d,
            input: -half,
        },
        Case {
            label: "pos_ratio_0.75",
            divisor: d,
            input: 3 * quarter,
        },
        Case {
            label: "neg_ratio_0.75",
            divisor: d,
            input: -3 * quarter,
        },
        Case {
            label: "pos_ratio_1.25",
            divisor: d,
            input: d + quarter,
        },
        Case {
            label: "neg_ratio_1.25",
            divisor: d,
            input: -(d + quarter),
        },
        Case {
            label: "pos_ratio_1.5_TIE",
            divisor: d,
            input: d + half,
        },
        Case {
            label: "neg_ratio_1.5_TIE",
            divisor: d,
            input: -(d + half),
        },
        Case {
            label: "pos_ratio_1.75",
            divisor: d,
            input: d + 3 * quarter,
        },
        Case {
            label: "neg_ratio_1.75",
            divisor: d,
            input: -(d + 3 * quarter),
        },
        Case {
            label: "pos_ratio_2.5_TIE",
            divisor: d,
            input: 2 * d + half,
        },
        Case {
            label: "neg_ratio_2.5_TIE",
            divisor: d,
            input: -(2 * d + half),
        },
    ]
}

fn large_magnitude_cases() -> Vec<(&'static str, IntegerRep)> {
    vec![
        ("i128_MAX", IntegerRep::MAX),
        ("i128_MAX_minus_1", IntegerRep::MAX - 1),
        ("i128_MIN_plus_1", IntegerRep::MIN + 1),
        ("large_2pow60", 1i128 << 60),
        ("large_2pow60_neg", -(1i128 << 60)),
        ("large_2pow90", 1i128 << 90),
        ("large_2pow90_neg", -(1i128 << 90)),
        ("large_2pow60_plus_999983", (1i128 << 60) + 999_983),
        ("large_2pow60_plus_999983_neg", -((1i128 << 60) + 999_983)),
    ]
}

fn main() {
    let divisors: Vec<IntegerRep> = vec![16, 8192, 16384];

    println!("divisor,input,label,q_old,q_new,equal,abs_over_2pow53");

    for &d in &divisors {
        for case in cases_for_divisor(d) {
            emit_row(case.divisor, case.input, case.label);
        }
        for &(label, input) in &large_magnitude_cases() {
            emit_row(d, input, label);
        }
    }
}

fn emit_row(divisor: IntegerRep, input: IntegerRep, label: &str) {
    let old = q_old(input, divisor);
    let over_2pow53 = input.unsigned_abs() > (1u128 << 53);
    match q_new(input, divisor) {
        Some(new) => println!(
            "{},{},{},{},{},{},{}",
            divisor,
            input,
            label,
            old,
            new,
            old == new,
            over_2pow53
        ),
        None => println!(
            "{},{},{},{},NA_unrepresentable_in_i128,NA,{}",
            divisor, input, label, old, over_2pow53
        ),
    }
}
