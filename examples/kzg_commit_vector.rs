use std::path::PathBuf;

use halo2_proofs::halo2curves::bn256::{Bn256, Fr};
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;

use ezkl::circuit::modules::polycommit::PolyCommitChip;
use ezkl::graph::{GraphCircuit, GraphSettings};
use ezkl::pfsys::srs::load_srs_prover;
use ezkl::pfsys::{load_vk, string_to_field};

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() {
    let settings_path = PathBuf::from(arg("--settings").expect("--settings is required"));
    let vk_path = PathBuf::from(arg("--vk").expect("--vk is required"));
    let srs_path = PathBuf::from(arg("--srs").expect("--srs is required"));
    let input_path = PathBuf::from(arg("--input").expect("--input is required"));

    let raw = std::fs::read_to_string(&input_path).expect("failed to read --input");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("--input is not valid JSON");
    let items = match &json {
        serde_json::Value::Array(a) => a.clone(),
        serde_json::Value::Object(o) => o
            .get("message")
            .and_then(|m| m.as_array())
            .cloned()
            .expect("object form must have a \"message\" array"),
        _ => panic!("--input must be a JSON array or an object with a \"message\" array"),
    };
    let message: Vec<Fr> = items
        .iter()
        .map(|v| {
            string_to_field::<Fr>(
                &v.as_str()
                    .expect("message entries must be hex strings")
                    .to_string(),
            )
        })
        .collect();

    let settings = GraphSettings::load(&settings_path).expect("failed to load settings");
    let srs = load_srs_prover::<KZGCommitmentScheme<Bn256>>(srs_path).expect("failed to load srs");
    let vk = load_vk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(vk_path, settings)
        .expect("failed to load vk");

    let points = PolyCommitChip::commit::<KZGCommitmentScheme<Bn256>>(
        message,
        (vk.cs().blinding_factors() + 1) as u32,
        &srs,
    );

    for p in points {
        let mut x = p.x.to_bytes();
        let mut y = p.y.to_bytes();
        x.reverse();
        y.reverse();
        println!("{}{}", hex::encode(x), hex::encode(y));
    }
}
