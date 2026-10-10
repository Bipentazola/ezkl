use crate::circuit::ops::poly::PolyOp;
use crate::circuit::*;
use crate::tensor::{DataFormat, KernelFormat};
use crate::tensor::{Tensor, TensorType, ValTensor, VarTensor};
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    dev::MockProver,
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::Fr as F;
use halo2curves::ff::{Field, PrimeField};
use itertools::Itertools;
#[cfg(not(any(
    all(target_arch = "wasm32", target_os = "unknown"),
    not(feature = "ezkl")
)))]
use ops::lookup::LookupOp;
use ops::region::RegionCtx;
use rand::rngs::OsRng;
use std::marker::PhantomData;

#[derive(Default)]
struct TestParams;

#[cfg(test)]
mod matmul {

    use super::*;

    const K: usize = 9;
    const LEN: usize = 3;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    #[test]
    fn matmulcircuit() {
        // parameters
        let mut a =
            Tensor::from((0..(LEN + 1) * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN + 1]).unwrap();

        let mut w = Tensor::from((0..LEN + 1).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN + 1, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod matmul_col_overflow_double_col {
    use super::*;

    const K: usize = 5;
    const LEN: usize = 6;
    const NUM_INNER_COLS: usize = 2;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, NUM_INNER_COLS, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn matmulcircuit() {
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod matmul_col_overflow {
    use super::*;

    const K: usize = 5;
    const LEN: usize = 6;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn matmulcircuit() {
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
#[cfg(all(
    feature = "ezkl",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
mod matmul_col_ultra_overflow_double_col {

    use halo2_proofs::poly::kzg::{
        commitment::KZGCommitmentScheme,
        multiopen::{ProverSHPLONK, VerifierSHPLONK},
        strategy::SingleStrategy,
    };
    use snark_verifier::system::halo2::transcript::evm::EvmTranscript;

    use super::*;

    const K: usize = 4;
    const LEN: usize = 10;
    const NUM_INNER_COLS: usize = 2;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, NUM_INNER_COLS, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn matmulcircuit() {
        // get some logs fam
        crate::logger::init_logger();
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let params = crate::pfsys::srs::gen_srs::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<_>,
        >(K as u32);

        let pk = crate::pfsys::create_keys::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            MatmulCircuit<F>,
        >(&circuit, &params, true)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit::<
            KZGCommitmentScheme<_>,
            _,
            ProverSHPLONK<_>,
            VerifierSHPLONK<_>,
            SingleStrategy<_>,
            _,
            EvmTranscript<_, _, _, _>,
            EvmTranscript<_, _, _, _>,
        >(
            circuit.clone(),
            vec![],
            &params,
            &pk,
            // use safe mode to verify that the proof is correct
            CheckMode::SAFE,
            None,
            None,
        );

        assert!(prover.is_ok());
    }
}

#[cfg(test)]
#[cfg(all(
    feature = "ezkl",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
mod matmul_col_ultra_overflow {

    use halo2_proofs::poly::kzg::{
        commitment::KZGCommitmentScheme,
        multiopen::{ProverSHPLONK, VerifierSHPLONK},
        strategy::SingleStrategy,
    };
    use itertools::Itertools;
    use snark_verifier::system::halo2::transcript::evm::EvmTranscript;

    use super::*;

    const K: usize = 4;
    const LEN: usize = 10;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn matmulcircuit() {
        // get some logs fam
        crate::logger::init_logger();
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let params = crate::pfsys::srs::gen_srs::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<_>,
        >(K as u32);

        let pk = crate::pfsys::create_keys::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            MatmulCircuit<F>,
        >(&circuit, &params, true)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit::<
            KZGCommitmentScheme<_>,
            _,
            ProverSHPLONK<_>,
            VerifierSHPLONK<_>,
            SingleStrategy<_>,
            _,
            EvmTranscript<_, _, _, _>,
            EvmTranscript<_, _, _, _>,
        >(
            circuit.clone(),
            vec![],
            &params,
            &pk,
            // use safe mode to verify that the proof is correct
            CheckMode::SAFE,
            None,
            None,
        );

        assert!(prover.is_ok());
    }
}

#[cfg(test)]
mod dot {
    use ops::poly::PolyOp;

    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod dot_col_overflow_triple_col {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 50;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            // used for constants in the padding
            let _fixed = cs.fixed_column();
            cs.enable_constant(_fixed);

            let a = VarTensor::new_advice(cs, K, 3, LEN);
            let b = VarTensor::new_advice(cs, K, 3, LEN);
            let output = VarTensor::new_advice(cs, K, 3, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 3, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod dot_col_overflow {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 50;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod sum {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Sum { axes: vec![0] }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn sumcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod sum_col_overflow_double_col {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 20;
    const NUM_INNER_COLS: usize = 2;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN);
            let b = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN);
            let output = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, NUM_INNER_COLS, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Sum { axes: vec![0] }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn sumcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod sum_col_overflow {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 20;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Sum { axes: vec![0] }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn sumcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod composition {

    use super::*;

    const K: usize = 9;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            // lots of stacked dot products
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        let _ = config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .unwrap();
                        let _ = config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .unwrap();
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod conv {

    use super::*;

    const K: usize = 22;
    const LEN: usize = 100;

    #[derive(Clone)]
    struct ConvCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: Vec<ValTensor<F>>,
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for ConvCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, (LEN + 1) * LEN);
            let b = VarTensor::new_advice(cs, K, 1, (LEN + 1) * LEN);
            let output = VarTensor::new_advice(cs, K, 1, (LEN + 1) * LEN);

            // column for constants
            let _constant = VarTensor::constant_cols(cs, K, 8, false);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Conv {
                                    padding: vec![(1, 1); 2],
                                    stride: vec![2; 2],
                                    group: 1,
                                    data_format: DataFormat::default(),
                                    kernel_format: KernelFormat::default(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn convcircuit() {
        // parameters
        let kernel_height = 2;
        let kernel_width = 3;
        let image_height = 5;
        let image_width = 7;
        let in_channels = 3;
        let out_channels = 2;

        let mut image =
            Tensor::from((0..in_channels * image_height * image_width).map(|_| F::random(OsRng)));
        image
            .reshape(&[1, in_channels, image_height, image_width])
            .unwrap();
        image.set_visibility(&crate::graph::Visibility::Private);

        let image = ValTensor::try_from(image).unwrap();

        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width })
                .map(|_| F::random(OsRng)),
        );
        kernels
            .reshape(&[out_channels, in_channels, kernel_height, kernel_width])
            .unwrap();
        kernels.set_visibility(&crate::graph::Visibility::Private);

        let kernels = ValTensor::try_from(kernels).unwrap();
        let mut bias = Tensor::from((0..{ out_channels }).map(|_| F::random(OsRng)));
        bias.set_visibility(&crate::graph::Visibility::Private);

        let bias = ValTensor::try_from(bias).unwrap();

        let circuit = ConvCircuit::<F> {
            inputs: [image, kernels, bias].to_vec(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn convcircuitnobias() {
        // parameters
        let kernel_height = 2;
        let kernel_width = 2;
        let image_height = 4;
        let image_width = 5;
        let in_channels = 3;
        let out_channels = 2;

        let mut image =
            Tensor::from((0..in_channels * image_height * image_width).map(|i| F::from(i as u64)));
        image
            .reshape(&[1, in_channels, image_height, image_width])
            .unwrap();
        image.set_visibility(&crate::graph::Visibility::Private);

        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width })
                .map(|i| F::from(i as u64)),
        );
        kernels
            .reshape(&[out_channels, in_channels, kernel_height, kernel_width])
            .unwrap();
        kernels.set_visibility(&crate::graph::Visibility::Private);

        let image = ValTensor::try_from(image).unwrap();
        let kernels = ValTensor::try_from(kernels).unwrap();

        let circuit = ConvCircuit::<F> {
            inputs: [image, kernels].to_vec(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
#[cfg(all(
    feature = "ezkl",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
mod conv_col_ultra_overflow {

    use halo2_proofs::poly::{
        kzg::strategy::SingleStrategy,
        kzg::{
            commitment::KZGCommitmentScheme,
            multiopen::{ProverSHPLONK, VerifierSHPLONK},
        },
    };
    use snark_verifier::system::halo2::transcript::evm::EvmTranscript;

    use super::*;

    const K: usize = 6;
    const LEN: usize = 10;

    #[derive(Clone)]
    struct ConvCircuit<F: PrimeField + TensorType + PartialOrd> {
        image: ValTensor<F>,
        kernel: ValTensor<F>,
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for ConvCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN * LEN);
            let _constant = VarTensor::constant_cols(cs, K, LEN * LEN * LEN * LEN, false);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &[&self.image, &self.kernel],
                                Box::new(PolyOp::Conv {
                                    padding: vec![(1, 1); 2],
                                    stride: vec![2; 2],
                                    group: 1,
                                    data_format: DataFormat::default(),
                                    kernel_format: KernelFormat::default(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn conv_circuit() {
        // parameters
        let kernel_height = 2;
        let kernel_width = 2;
        let image_height = LEN;
        let image_width = LEN;
        let in_channels = 3;
        let out_channels = 2;

        // get some logs fam
        crate::logger::init_logger();
        let mut image =
            Tensor::from((0..in_channels * image_height * image_width).map(|i| F::from(i as u64)));
        image
            .reshape(&[1, in_channels, image_height, image_width])
            .unwrap();
        image.set_visibility(&crate::graph::Visibility::Private);

        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width })
                .map(|i| F::from(i as u64)),
        );
        kernels
            .reshape(&[out_channels, in_channels, kernel_height, kernel_width])
            .unwrap();
        kernels.set_visibility(&crate::graph::Visibility::Private);

        let circuit = ConvCircuit::<F> {
            image: ValTensor::try_from(image).unwrap(),
            kernel: ValTensor::try_from(kernels).unwrap(),
            _marker: PhantomData,
        };

        let params = crate::pfsys::srs::gen_srs::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<_>,
        >(K as u32);

        let pk = crate::pfsys::create_keys::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            ConvCircuit<F>,
        >(&circuit, &params, true)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit::<
            KZGCommitmentScheme<_>,
            _,
            ProverSHPLONK<_>,
            VerifierSHPLONK<_>,
            SingleStrategy<_>,
            _,
            EvmTranscript<_, _, _, _>,
            EvmTranscript<_, _, _, _>,
        >(
            circuit.clone(),
            vec![],
            &params,
            &pk,
            // use safe mode to verify that the proof is correct
            CheckMode::SAFE,
            None,
            None,
        );

        assert!(prover.is_ok());
    }
}

#[cfg(test)]
// not wasm 32 unknown
#[cfg(all(
    feature = "ezkl",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
mod conv_relu_col_ultra_overflow {

    use halo2_proofs::poly::kzg::{
        commitment::KZGCommitmentScheme,
        multiopen::{ProverSHPLONK, VerifierSHPLONK},
        strategy::SingleStrategy,
    };
    use snark_verifier::system::halo2::transcript::evm::EvmTranscript;

    use super::*;

    const K: usize = 8;
    const LEN: usize = 15;

    #[derive(Clone)]
    struct ConvCircuit<F: PrimeField + TensorType + PartialOrd> {
        image: ValTensor<F>,
        kernel: ValTensor<F>,
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for ConvCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN * 4);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN * 4);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN * 4);
            let mut base_config =
                Self::Config::configure(cs, &[a.clone(), b.clone()], &output, CheckMode::SAFE);
            // sets up a new relu table

            base_config
                .configure_range_check(cs, &a, &b, (-1, 1), K)
                .unwrap();

            base_config
                .configure_range_check(cs, &a, &b, (0, 1), K)
                .unwrap();

            let _constant = VarTensor::constant_cols(cs, K, 8, false);

            base_config.clone()
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.layout_range_checks(&mut layouter).unwrap();
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 2, 2);
                        let output = config
                            .layout(
                                &mut region,
                                &[&self.image, &self.kernel],
                                Box::new(PolyOp::Conv {
                                    padding: vec![(1, 1); 2],
                                    stride: vec![2; 2],
                                    group: 1,
                                    data_format: DataFormat::default(),
                                    kernel_format: KernelFormat::default(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis);
                        let _output = config
                            .layout(
                                &mut region,
                                &[&output.unwrap().unwrap()],
                                Box::new(PolyOp::LeakyReLU {
                                    slope: 0.0.into(),
                                    scale: 1,
                                }),
                            )
                            .unwrap();
                        Ok(())
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn conv_relu_circuit() {
        // parameters
        let kernel_height = 2;
        let kernel_width = 2;
        let image_height = LEN;
        let image_width = LEN;
        let in_channels = 3;
        let out_channels = 2;

        // get some logs fam
        crate::logger::init_logger();
        let mut image =
            Tensor::from((0..in_channels * image_height * image_width).map(|_| F::from(0)));
        image
            .reshape(&[1, in_channels, image_height, image_width])
            .unwrap();
        image.set_visibility(&crate::graph::Visibility::Private);

        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width }).map(|_| F::from(0)),
        );
        kernels
            .reshape(&[out_channels, in_channels, kernel_height, kernel_width])
            .unwrap();
        kernels.set_visibility(&crate::graph::Visibility::Private);

        let circuit = ConvCircuit::<F> {
            image: ValTensor::try_from(image).unwrap(),
            kernel: ValTensor::try_from(kernels).unwrap(),
            _marker: PhantomData,
        };

        let params = crate::pfsys::srs::gen_srs::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<_>,
        >(K as u32);

        let pk = crate::pfsys::create_keys::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            ConvCircuit<F>,
        >(&circuit, &params, true)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit::<
            KZGCommitmentScheme<_>,
            _,
            ProverSHPLONK<_>,
            VerifierSHPLONK<_>,
            SingleStrategy<_>,
            _,
            EvmTranscript<_, _, _, _>,
            EvmTranscript<_, _, _, _>,
        >(
            circuit.clone(),
            vec![],
            &params,
            &pk,
            CheckMode::SAFE,
            // use safe mode to verify that the proof is correct
            None,
            None,
        );

        assert!(prover.is_ok());
    }
}

#[cfg(test)]
mod add_w_shape_casting {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Add),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn addcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..1).map(|i| Value::known(F::from(i + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod add {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Add),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn addcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod dynamic_lookup {
    use super::*;

    const K: usize = 6;
    const LEN: usize = 4;
    const NUM_LOOP: usize = 5;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        tables: [[ValTensor<F>; 2]; NUM_LOOP],
        lookups: [[ValTensor<F>; 2]; NUM_LOOP],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 2, LEN);
            let b = VarTensor::new_advice(cs, K, 2, LEN);
            let c: VarTensor = VarTensor::new_advice(cs, K, 2, LEN);

            let d = VarTensor::new_advice(cs, K, 1, LEN);
            let e = VarTensor::new_advice(cs, K, 1, LEN);
            let f: VarTensor = VarTensor::new_advice(cs, K, 1, LEN);

            let _constant = VarTensor::constant_cols(cs, K, LEN * NUM_LOOP, false);

            let mut config =
                Self::Config::configure(cs, &[a.clone(), b.clone()], &c, CheckMode::SAFE);
            config
                .configure_dynamic_lookup(
                    cs,
                    &[a.clone(), b.clone(), c.clone()],
                    &[d.clone(), e.clone(), f.clone()],
                )
                .unwrap();
            config
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        for i in 0..NUM_LOOP {
                            layouts::dynamic_lookup(
                                &config,
                                &mut region,
                                &self.lookups[i].iter().collect_vec().try_into().unwrap(),
                                &self.tables[i].iter().collect_vec().try_into().unwrap(),
                            )
                            .map_err(|_| Error::Synthesis)?;
                        }
                        assert_eq!(
                            region.dynamic_lookup_col_coord(),
                            NUM_LOOP * self.tables[0][0].len()
                        );
                        assert_eq!(region.dynamic_lookup_index(), NUM_LOOP);

                        Ok(())
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn dynamiclookupcircuit() {
        // parameters
        let tables = (0..NUM_LOOP)
            .map(|loop_idx| {
                [
                    ValTensor::from(Tensor::from(
                        (0..LEN).map(|i| Value::known(F::from((i * loop_idx) as u64 + 1))),
                    )),
                    ValTensor::from(Tensor::from(
                        (0..LEN).map(|i| Value::known(F::from((loop_idx * i * i) as u64 + 1))),
                    )),
                ]
            })
            .collect::<Vec<_>>();

        let lookups = (0..NUM_LOOP)
            .map(|loop_idx| {
                [
                    ValTensor::from(Tensor::from(
                        (0..3).map(|i| Value::known(F::from((i * loop_idx) as u64 + 1))),
                    )),
                    ValTensor::from(Tensor::from(
                        (0..3).map(|i| Value::known(F::from((loop_idx * i * i) as u64 + 1))),
                    )),
                ]
            })
            .collect::<Vec<_>>();

        let circuit = MyCircuit::<F> {
            tables: tables.clone().try_into().unwrap(),
            lookups: lookups.try_into().unwrap(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();

        let lookups = (0..NUM_LOOP)
            .map(|loop_idx| {
                let prev_idx = if loop_idx == 0 {
                    NUM_LOOP - 1
                } else {
                    loop_idx - 1
                };
                [
                    ValTensor::from(Tensor::from(
                        (0..3).map(|i| Value::known(F::from((i * prev_idx) as u64 + 1))),
                    )),
                    ValTensor::from(Tensor::from(
                        (0..3).map(|i| Value::known(F::from((prev_idx * i * i) as u64 + 1))),
                    )),
                ]
            })
            .collect::<Vec<_>>();

        let circuit = MyCircuit::<F> {
            tables: tables.try_into().unwrap(),
            lookups: lookups.try_into().unwrap(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }
}

#[cfg(test)]
mod shuffle {
    use super::*;

    const K: usize = 6;
    const LEN: usize = 4;
    const NUM_LOOP: usize = 5;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; NUM_LOOP],
        references: [ValTensor<F>; NUM_LOOP],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 2, LEN);
            let b = VarTensor::new_advice(cs, K, 2, LEN);
            let c: VarTensor = VarTensor::new_advice(cs, K, 2, LEN);

            let d = VarTensor::new_advice(cs, K, 1, LEN);
            let e = VarTensor::new_advice(cs, K, 1, LEN);
            let f: VarTensor = VarTensor::new_advice(cs, K, 1, LEN);

            let _constant = VarTensor::constant_cols(cs, K, LEN * NUM_LOOP, false);

            let mut config =
                Self::Config::configure(cs, &[a.clone(), b.clone()], &c, CheckMode::SAFE);
            config
                .configure_shuffles(
                    cs,
                    &[a.clone(), b.clone(), c.clone()],
                    &[d.clone(), e.clone(), f.clone()],
                )
                .unwrap();
            config
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        for i in 0..NUM_LOOP {
                            layouts::shuffles(
                                &config,
                                &mut region,
                                &[&self.inputs[i]],
                                &[&self.references[i]],
                                layouts::SortCollisionMode::Unsorted,
                            )
                            .map_err(|_| Error::Synthesis)?;
                        }
                        assert_eq!(
                            region.shuffle_col_coord(),
                            NUM_LOOP * self.references[0].len()
                        );
                        assert_eq!(region.shuffle_index(), NUM_LOOP);

                        Ok(())
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn shufflecircuit() {
        // parameters
        let references = (0..NUM_LOOP)
            .map(|loop_idx| {
                ValTensor::from(Tensor::from(
                    (0..LEN).map(|i| Value::known(F::from((i * loop_idx) as u64 + 1))),
                ))
            })
            .collect::<Vec<_>>();

        let inputs = (0..NUM_LOOP)
            .map(|loop_idx| {
                ValTensor::from(Tensor::from(
                    (0..LEN)
                        .rev()
                        .map(|i| Value::known(F::from((i * loop_idx) as u64 + 1))),
                ))
            })
            .collect::<Vec<_>>();

        let circuit = MyCircuit::<F> {
            references: references.clone().try_into().unwrap(),
            inputs: inputs.try_into().unwrap(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();

        let inputs = (0..NUM_LOOP)
            .map(|loop_idx| {
                let prev_idx = if loop_idx == 0 {
                    NUM_LOOP - 1
                } else {
                    loop_idx - 1
                };
                ValTensor::from(Tensor::from(
                    (0..LEN)
                        .rev()
                        .map(|i| Value::known(F::from((i * prev_idx) as u64 + 1))),
                ))
            })
            .collect::<Vec<_>>();

        let circuit = MyCircuit::<F> {
            references: references.try_into().unwrap(),
            inputs: inputs.try_into().unwrap(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }
}

#[cfg(test)]
mod add_with_overflow {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 50;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Add),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn addcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod add_with_overflow_and_poseidon {
    use std::collections::HashMap;

    use halo2curves::bn256::Fr;

    use crate::circuit::modules::{
        poseidon::{
            spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH},
            PoseidonChip, PoseidonConfig,
        },
        Module, ModulePlanner,
    };

    use super::*;

    const K: usize = 15;
    const LEN: usize = 50;
    const WIDTH: usize = POSEIDON_WIDTH;
    const RATE: usize = POSEIDON_RATE;

    #[derive(Debug, Clone)]
    struct MyCircuitConfig {
        base: BaseConfig<Fr>,
        poseidon: PoseidonConfig<WIDTH, RATE>,
    }

    #[derive(Clone)]
    struct MyCircuit {
        inputs: [ValTensor<Fr>; 2],
    }

    impl Circuit<Fr> for MyCircuit {
        type Config = MyCircuitConfig;
        type FloorPlanner = ModulePlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            let base = BaseConfig::configure(cs, &[a, b], &output, CheckMode::SAFE);
            VarTensor::constant_cols(cs, K, 2, false);

            let poseidon = PoseidonChip::<PoseidonSpec, WIDTH, RATE>::configure(cs, ());

            MyCircuitConfig { base, poseidon }
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            let poseidon_chip: PoseidonChip<PoseidonSpec, WIDTH, RATE> =
                PoseidonChip::new(config.poseidon.clone());

            let assigned_inputs_a =
                poseidon_chip.layout(&mut layouter, &self.inputs[0..1], 0, &mut HashMap::new())?;
            let assigned_inputs_b =
                poseidon_chip.layout(&mut layouter, &self.inputs[1..2], 1, &mut HashMap::new())?;

            layouter.assign_region(|| "_new_module", |_| Ok(()))?;

            let inputs = vec![&assigned_inputs_a, &assigned_inputs_b];

            layouter.assign_region(
                || "model",
                |region| {
                    let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                    config
                        .base
                        .layout(&mut region, &inputs, Box::new(PolyOp::Add))
                        .map_err(|_| Error::Synthesis)
                },
            )?;

            Ok(())
        }
    }

    #[test]
    fn addcircuit() {
        let a = (0..LEN)
            .map(|i| halo2curves::bn256::Fr::from(i as u64 + 1))
            .collect::<Vec<_>>();
        let b = (0..LEN)
            .map(|i| halo2curves::bn256::Fr::from(i as u64 + 1))
            .collect::<Vec<_>>();
        let commitment_a = PoseidonChip::<PoseidonSpec, WIDTH, RATE>::run(a.clone()).unwrap()[0][0];

        let commitment_b = PoseidonChip::<PoseidonSpec, WIDTH, RATE>::run(b.clone()).unwrap()[0][0];

        // parameters
        let a = Tensor::from(a.into_iter().map(Value::known));
        let b = Tensor::from(b.into_iter().map(Value::known));
        let circuit = MyCircuit {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
        };

        let prover =
            MockProver::run(K as u32, &circuit, vec![vec![commitment_a, commitment_b]]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn addcircuit_bad_hashes() {
        let a = (0..LEN)
            .map(|i| halo2curves::bn256::Fr::from(i as u64 + 1))
            .collect::<Vec<_>>();
        let b = (0..LEN)
            .map(|i| halo2curves::bn256::Fr::from(i as u64 + 1))
            .collect::<Vec<_>>();
        let commitment_a =
            PoseidonChip::<PoseidonSpec, WIDTH, RATE>::run(a.clone()).unwrap()[0][0] + Fr::one();

        let commitment_b =
            PoseidonChip::<PoseidonSpec, WIDTH, RATE>::run(b.clone()).unwrap()[0][0] + Fr::one();

        // parameters
        let a = Tensor::from(a.into_iter().map(Value::known));
        let b = Tensor::from(b.into_iter().map(Value::known));
        let circuit = MyCircuit {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
        };

        let prover =
            MockProver::run(K as u32, &circuit, vec![vec![commitment_a, commitment_b]]).unwrap();
        assert!(prover.verify().is_err());
    }
}

#[cfg(test)]
mod sub {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Sub),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn subcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod mult {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Mult),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn multcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod pow {
    use super::*;

    const K: usize = 8;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.iter().collect_vec(),
                                Box::new(PolyOp::Pow(5)),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn powcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod matmul_relu {
    use super::*;

    const K: usize = 18;
    const LEN: usize = 32;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    // A columnar ReLu MLP
    #[derive(Clone)]
    struct MyConfig<F: PrimeField + TensorType + PartialOrd> {
        base_config: BaseConfig<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = MyConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            let mut base_config =
                BaseConfig::configure(cs, &[a.clone(), b.clone()], &output, CheckMode::SAFE);

            base_config
                .configure_range_check(cs, &a, &b, (-1, 1), K)
                .unwrap();

            base_config
                .configure_range_check(cs, &a, &b, (0, 1023), K)
                .unwrap();

            let _constant = VarTensor::constant_cols(cs, K, 8, false);

            MyConfig { base_config }
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config
                .base_config
                .layout_range_checks(&mut layouter)
                .unwrap();
            layouter.assign_region(
                || "",
                |region| {
                    let mut region = RegionCtx::new(region, 0, 1, 1024, 2);
                    let op = PolyOp::Einsum {
                        equation: "ij,jk->ik".to_string(),
                    };
                    let output = config
                        .base_config
                        .layout(&mut region, &self.inputs.iter().collect_vec(), Box::new(op))
                        .unwrap();
                    let _output = config
                        .base_config
                        .layout(
                            &mut region,
                            &[&output.unwrap()],
                            Box::new(PolyOp::LeakyReLU {
                                slope: 0.0.into(),
                                scale: 1,
                            }),
                        )
                        .unwrap();
                    Ok(())
                },
            )?;

            Ok(())
        }
    }

    #[test]
    fn matmulrelucircuit() {
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|_| Value::known(F::from(1))));
        a.reshape(&[LEN, LEN]).unwrap();

        // parameters
        let mut b = Tensor::from((0..LEN).map(|_| Value::known(F::from(1))));
        b.reshape(&[LEN, 1]).unwrap();

        let circuit = MyCircuit {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod relu {
    use super::*;
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };

    const K: u32 = 8;

    #[derive(Clone)]
    struct ReLUCircuit<F: PrimeField + TensorType + PartialOrd> {
        pub input: ValTensor<F>,
    }

    impl Circuit<F> for ReLUCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let advices = (0..3)
                .map(|_| VarTensor::new_advice(cs, 8, 1, 3))
                .collect::<Vec<_>>();

            let mut config = BaseConfig::configure(
                cs,
                &[advices[0].clone(), advices[1].clone()],
                &advices[2],
                CheckMode::SAFE,
            );

            config
                .configure_range_check(cs, &advices[0], &advices[1], (-1, 1), K as usize)
                .unwrap();

            config
                .configure_range_check(cs, &advices[0], &advices[1], (0, 1), K as usize)
                .unwrap();

            let _constant = VarTensor::constant_cols(cs, K as usize, 8, false);

            config
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>, // layouter is our 'write buffer' for the circuit
        ) -> Result<(), Error> {
            config.layout_range_checks(&mut layouter).unwrap();
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 2, 2);
                        Ok(config
                            .layout(
                                &mut region,
                                &[&self.input],
                                Box::new(PolyOp::LeakyReLU {
                                    slope: 0.0.into(),
                                    scale: 1,
                                }),
                            )
                            .unwrap())
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    #[test]
    fn relucircuit() {
        let input: Tensor<Value<F>> =
            Tensor::new(Some(&[Value::<F>::known(F::from(1_u64)); 4]), &[4]).unwrap();

        let circuit = ReLUCircuit::<F> {
            input: ValTensor::from(input),
        };

        let prover = MockProver::run(K, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod div {
    use super::*;
    use crate::circuit::ops::base::BaseOp;
    use crate::circuit::ops::layouts;
    use crate::fieldutils::integer_rep_to_felt;
    use crate::tensor::create_constant_tensor;
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };

    const K: u32 = 13;
    const BASE: usize = 16;
    const LEGS: usize = 4;

    #[derive(Clone)]
    struct DivCircuit<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> {
        input: ValTensor<F>,
        divisor: F,
    }

    impl Circuit<F> for DivCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let advices = (0..3)
                .map(|_| VarTensor::new_advice(cs, K as usize, 1, 1024))
                .collect::<Vec<_>>();

            let mut config = BaseConfig::configure(
                cs,
                &[advices[0].clone(), advices[1].clone()],
                &advices[2],
                CheckMode::SAFE,
            );

            config
                .configure_range_check(cs, &advices[0], &advices[1], (-1, 1), K as usize)
                .unwrap();
            config
                .configure_range_check(
                    cs,
                    &advices[0],
                    &advices[1],
                    (0, (BASE - 1) as i128),
                    K as usize,
                )
                .unwrap();

            let _constant = VarTensor::constant_cols(cs, K as usize, 8, false);

            config
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.layout_range_checks(&mut layouter).unwrap();
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, BASE, LEGS);
                        Ok(
                            layouts::div(&config, &mut region, &[&self.input], self.divisor)
                                .unwrap(),
                        )
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    fn run_div_case(divisor_int: i128, inputs: &[i128]) {
        let input: Tensor<Value<F>> = Tensor::new(
            Some(
                &inputs
                    .iter()
                    .map(|&x| Value::known(integer_rep_to_felt(x)))
                    .collect::<Vec<_>>(),
            ),
            &[inputs.len()],
        )
        .unwrap();

        let circuit = DivCircuit::<F> {
            input: ValTensor::from(input),
            divisor: integer_rep_to_felt(divisor_int),
        };

        let prover = MockProver::run(K, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn div_baseline_positive() {
        run_div_case(
            16,
            &[
                0, 80, -80, 1, -1, 15, -15, 16, -16, 17, -17, 53, -53, 4, -4, 8, -8, 12, -12, 20,
                -20, 24, -24, 28, -28, 40, -40,
            ],
        );
        run_div_case(
            8192,
            &[
                0, 40960, -40960, 1, -1, 8191, -8191, 8192, -8192, 8193, -8193, 27306, -27306,
                2048, -2048, 4096, -4096, 6144, -6144, 10240, -10240, 12288, -12288, 14336, -14336,
                20480, -20480,
            ],
        );
        run_div_case(
            16384,
            &[
                0, 81920, -81920, 1, -1, 16383, -16383, 16384, -16384, 16385, -16385, 54613,
                -54613, 4096, -4096, 8192, -8192, 12288, -12288, 20480, -20480, 24576, -24576,
                28672, -28672, 40960, -40960,
            ],
        );
    }

    #[derive(Clone)]
    struct BadDivWitnessCircuit<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> {
        input: ValTensor<F>,
        claimed_q: ValTensor<F>,
        divisor: F,
    }

    impl Circuit<F> for BadDivWitnessCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            DivCircuit::<F>::configure(cs)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.layout_range_checks(&mut layouter).unwrap();
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, BASE, LEGS);
                        let divisor_tensor = create_constant_tensor(self.divisor, 1);
                        let (base, legs) = (region.base(), region.legs());

                        let claimed_output = layouts::decompose(
                            &config,
                            &mut region,
                            &[&self.claimed_q],
                            &base,
                            &legs,
                            false,
                        )
                        .unwrap()
                        .1;

                        let product = layouts::pairwise(
                            &config,
                            &mut region,
                            &[&claimed_output, &divisor_tensor],
                            BaseOp::Mult,
                        )
                        .unwrap();

                        layouts::diff_less_than(
                            &config,
                            &mut region,
                            &[&self.input, &product],
                            self.divisor,
                        )
                        .unwrap();

                        Ok(())
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    fn bad_witness_accepted(divisor_int: i128, input_int: i128, bad_q_int: i128) -> bool {
        let input: Tensor<Value<F>> =
            Tensor::new(Some(&[Value::known(integer_rep_to_felt(input_int))]), &[1]).unwrap();
        let claimed_q: Tensor<Value<F>> =
            Tensor::new(Some(&[Value::known(integer_rep_to_felt(bad_q_int))]), &[1]).unwrap();

        let circuit = BadDivWitnessCircuit::<F> {
            input: ValTensor::from(input),
            claimed_q: ValTensor::from(claimed_q),
            divisor: integer_rep_to_felt(divisor_int),
        };

        let prover = MockProver::run(K, &circuit, vec![]).unwrap();
        prover.verify().is_ok()
    }

    #[test]
    fn div_negative_bad_quotient_rejected() {
        assert!(bad_witness_accepted(16, 53, 3));

        assert!(!bad_witness_accepted(16, 53, 6));

        assert!(!bad_witness_accepted(16, 53, 0));

        assert!(!bad_witness_accepted(16, -53, 3));

        assert!(!bad_witness_accepted(8192, 27306, 5));
    }
}

#[cfg(test)]
mod div_round_nearest {
    use super::*;
    use crate::circuit::ops::base::BaseOp;
    use crate::circuit::ops::layouts;
    use crate::fieldutils::{felt_to_integer_rep, integer_rep_to_felt};
    use crate::tensor::{create_constant_tensor, create_zero_tensor};
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };

    const K: u32 = 15;
    const BASE: usize = 16384;
    const LEGS: usize = 2;

    fn configure_shared(cs: &mut ConstraintSystem<F>) -> BaseConfig<F> {
        let advices = (0..3)
            .map(|_| VarTensor::new_advice(cs, K as usize, 1, 1024))
            .collect::<Vec<_>>();

        let mut config = BaseConfig::configure(
            cs,
            &[advices[0].clone(), advices[1].clone()],
            &advices[2],
            CheckMode::SAFE,
        );

        config
            .configure_range_check(cs, &advices[0], &advices[1], (-1, 1), K as usize)
            .unwrap();
        config
            .configure_range_check(
                cs,
                &advices[0],
                &advices[1],
                (0, (BASE - 1) as i128),
                K as usize,
            )
            .unwrap();
        config
            .configure_range_check(cs, &advices[0], &advices[1], (0, BASE as i128), K as usize)
            .unwrap();

        let _constant = VarTensor::constant_cols(cs, K as usize, 8, false);

        config
    }

    fn single_value<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
        x: F,
    ) -> ValTensor<F> {
        Tensor::new(Some(&[Value::known(x)]), &[1]).unwrap().into()
    }

    #[derive(Clone)]
    struct DivRoundNearestCircuit<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> {
        input: ValTensor<F>,
        divisor: F,
    }

    impl Circuit<F> for DivRoundNearestCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            configure_shared(cs)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.layout_range_checks(&mut layouter).unwrap();
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, BASE, LEGS);
                        Ok(layouts::div_round_nearest(
                            &config,
                            &mut region,
                            &[&self.input],
                            self.divisor,
                        )
                        .unwrap())
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    fn run_positive_case(divisor_int: i128, inputs: &[i128]) {
        let input: Tensor<Value<F>> = Tensor::new(
            Some(
                &inputs
                    .iter()
                    .map(|&x| Value::known(integer_rep_to_felt(x)))
                    .collect::<Vec<_>>(),
            ),
            &[inputs.len()],
        )
        .unwrap();

        let circuit = DivRoundNearestCircuit::<F> {
            input: ValTensor::from(input),
            divisor: integer_rep_to_felt(divisor_int),
        };

        let prover = MockProver::run(K, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    fn positive_matrix_inputs(d: i128) -> Vec<(&'static str, i128)> {
        let half = d / 2;
        vec![
            ("zero", 0),
            ("pos_multiple_3x", 3 * d),
            ("neg_multiple_3x", -3 * d),
            ("pos_normal", 2 * d + 5),
            ("neg_normal", -(2 * d + 5)),
            ("pos_tie_0.5", half),
            ("neg_tie_0.5", -half),
            ("pos_tie_1.5", d + half),
            ("neg_tie_1.5", -(d + half)),
            ("pos_tie_2.5", 2 * d + half),
            ("neg_tie_2.5", -(2 * d + half)),
            ("pos_near_bottom_s1", 2 * d + (-half + 1)),
            ("neg_near_bottom_s1", -2 * d + (-half + 1)),
            ("pos_near_top_sDminus1", 2 * d + (half - 1)),
            ("neg_near_top_sDminus1", -2 * d + (half - 1)),
            ("pos_large_q_max", 65535 * d),
            ("neg_large_q_max", -65535 * d),
            ("pos_large_q_max_nonzero_r", 65534 * d + (half - 1)),
            ("neg_large_q_max_nonzero_r", -65534 * d - (half - 1)),
        ]
    }

    #[test]
    fn positive_matrix_all_divisors() {
        for &d in &[16i128, 8192, 16384] {
            let cases = positive_matrix_inputs(d);
            let inputs = cases.iter().map(|&(_, x)| x).collect::<Vec<_>>();
            run_positive_case(d, &inputs);
        }
    }

    const WIDE_DIVISORS: [i128; 2] = [1 << 21, 1 << 28];

    #[test]
    fn limbs_for_bound_picks_the_smallest_sufficient_limb_count() {
        let base = BASE as i128;
        assert_eq!(layouts::limbs_for_bound(base - 1, base).unwrap(), 1);
        assert_eq!(layouts::limbs_for_bound(base, base).unwrap(), 2);
        assert_eq!(layouts::limbs_for_bound(1 << 21, base).unwrap(), 2);
        assert_eq!(layouts::limbs_for_bound(base * base - 1, base).unwrap(), 2);
        assert_eq!(layouts::limbs_for_bound(base * base, base).unwrap(), 3);
        assert_eq!(layouts::limbs_for_bound(1 << 28, base).unwrap(), 3);

        for &d in WIDE_DIVISORS.iter() {
            let n = layouts::limbs_for_bound(d, base).unwrap();
            assert!(base.pow(n as u32) - 1 >= d);
            assert!(base.pow(n as u32 - 1) - 1 < d, "n is not minimal for {d}");
        }
    }

    #[test]
    fn positive_matrix_wide_divisors() {
        for &d in WIDE_DIVISORS.iter() {
            let cases = positive_matrix_inputs(d);
            let inputs = cases.iter().map(|&(_, x)| x).collect::<Vec<_>>();
            run_positive_case(d, &inputs);
        }
    }

    #[test]
    fn wide_path_row_cost_report() {
        use crate::circuit::ops::region::RegionSettings;
        use halo2_proofs::plonk::ConstraintSystem;

        const N_ELEMENTS: usize = 512;

        let rows_per_element = |divisor_int: i128| -> f64 {
            let mut cs = ConstraintSystem::<F>::default();
            let advices = (0..3)
                .map(|_| VarTensor::new_advice(&mut cs, K as usize, 1, 1024))
                .collect::<Vec<_>>();
            let config = BaseConfig::configure(
                &mut cs,
                &[advices[0].clone(), advices[1].clone()],
                &advices[2],
                CheckMode::SAFE,
            );

            let input: Tensor<Value<F>> = Tensor::new(
                Some(
                    &(0..N_ELEMENTS)
                        .map(|i| Value::known(integer_rep_to_felt((i as i128) * divisor_int + 7)))
                        .collect::<Vec<_>>(),
                ),
                &[N_ELEMENTS],
            )
            .unwrap();
            let input: ValTensor<F> = input.into();

            let mut region =
                RegionCtx::new_dummy(0, 2, RegionSettings::new(true, false, BASE, LEGS));
            let before = region.row();
            layouts::div_round_nearest(
                &config,
                &mut region,
                &[&input],
                integer_rep_to_felt(divisor_int),
            )
            .unwrap();
            (region.row() - before) as f64 / N_ELEMENTS as f64
        };

        println!("\n--- div_round_nearest rows/element by divisor ---");
        let mut measured = vec![];
        for &(label, d) in &[
            ("128   (ordinary rescale, narrow)", 1i128 << 7),
            ("16384 (= decomp_base, gap check skipped)", 1 << 14),
            ("2^21  (3 fused 2^7 rescales, wide n=2)", 1 << 21),
            ("2^28  (wide n=3)", 1 << 28),
        ] {
            let r = rows_per_element(d);
            println!("{label:<44} {r:>8.3} rows/element");
            measured.push((d, r));
        }

        let get = |d: i128| measured.iter().find(|&&(dd, _)| dd == d).unwrap().1;

        assert!(get(1 << 14) < get(1 << 7));
        assert!(get(1 << 21) > get(1 << 14));
        assert!(get(1 << 28) > get(1 << 21));
        let narrow = get(1 << 7);
        let widening_cost = get(1 << 21) - narrow;
        let net = 2.0 * narrow - widening_cost;
        println!(
            "\nfusing 3x 2^7 -> 1x 2^21: deletes 2 x {narrow:.1} = {:.1}, \
             pays {widening_cost:.1} to widen the survivor, net {net:.1} \
             rows/element saved ({:.0}% of the deleted nodes' cost kept)",
            2.0 * narrow,
            100.0 * net / (2.0 * narrow),
        );
        assert!(
            net > 0.0,
            "widening costs {widening_cost:.3} rows/element against {:.3} saved by \
             deleting the fused-away nodes -- fusion would increase the row count",
            2.0 * narrow,
        );
    }

    #[derive(Clone)]
    struct DivRoundNearestDecomposeTablesOnlyCircuit<
        F: PrimeField + TensorType + PartialOrd + std::hash::Hash,
    > {
        input: ValTensor<F>,
        divisor: F,
    }

    impl Circuit<F> for DivRoundNearestDecomposeTablesOnlyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let advices = (0..3)
                .map(|_| VarTensor::new_advice(cs, K as usize, 1, 1024))
                .collect::<Vec<_>>();

            let mut config = BaseConfig::configure(
                cs,
                &[advices[0].clone(), advices[1].clone()],
                &advices[2],
                CheckMode::SAFE,
            );

            config
                .configure_range_check(cs, &advices[0], &advices[1], (-1, 1), K as usize)
                .unwrap();
            config
                .configure_range_check(
                    cs,
                    &advices[0],
                    &advices[1],
                    (0, (BASE - 1) as i128),
                    K as usize,
                )
                .unwrap();

            let _constant = VarTensor::constant_cols(cs, K as usize, 8, false);

            config
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.layout_range_checks(&mut layouter).unwrap();
            layouter.assign_region(
                || "",
                |region| {
                    let mut region = RegionCtx::new(region, 0, 1, BASE, LEGS);
                    layouts::div_round_nearest(&config, &mut region, &[&self.input], self.divisor)
                        .map(|_| ())
                        .map_err(|_| Error::Synthesis)
                },
            )?;

            Ok(())
        }
    }

    #[test]
    fn wide_path_adds_no_range_check_table() {
        let run = |divisor_int: i128| -> bool {
            let prev_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let circuit = DivRoundNearestDecomposeTablesOnlyCircuit::<F> {
                    input: single_value(integer_rep_to_felt(3 * divisor_int + 7)),
                    divisor: integer_rep_to_felt(divisor_int),
                };
                match MockProver::run(K, &circuit, vec![]) {
                    Ok(prover) => prover.verify().is_ok(),
                    Err(_) => false,
                }
            }));
            std::panic::set_hook(prev_hook);
            outcome.unwrap_or(false)
        };

        assert!(
            !run(16),
            "control failed: the narrow path was accepted without its (0, BASE) table, \
             so this test cannot distinguish 'needs no new table' from 'tables not enforced'"
        );

        for &d in WIDE_DIVISORS.iter() {
            assert!(
                run(d),
                "wide divisor {d} needs a range-check table beyond decompose's own"
            );
        }
    }

    #[test]
    fn positive_matrix_multi_dim_tensor() {
        for &d in &[16i128, 8192, 16384] {
            let cases = positive_matrix_inputs(d);
            let mut inputs = cases.iter().map(|&(_, x)| x).collect::<Vec<_>>();
            inputs.push(0);
            assert_eq!(inputs.len(), 20);
            let shape = [1, 4, 5];

            let input: Tensor<Value<F>> = Tensor::new(
                Some(
                    &inputs
                        .iter()
                        .map(|&x| Value::known(integer_rep_to_felt(x)))
                        .collect::<Vec<_>>(),
                ),
                &shape,
            )
            .unwrap();

            let circuit = DivRoundNearestCircuit::<F> {
                input: ValTensor::from(input),
                divisor: integer_rep_to_felt(d),
            };

            let prover = MockProver::run(K, &circuit, vec![]).unwrap();
            prover.assert_satisfied();
        }
    }

    #[derive(Clone)]
    struct BadDivRoundNearestCircuit<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> {
        input: ValTensor<F>,
        claimed_q: ValTensor<F>,
        claimed_r: ValTensor<F>,
        divisor: F,
    }

    impl Circuit<F> for BadDivRoundNearestCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            configure_shared(cs)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.layout_range_checks(&mut layouter).unwrap();
            layouter.assign_region(
                || "",
                |region| {
                    let body = || -> Result<(), crate::circuit::CircuitError> {
                        let mut region = RegionCtx::new(region, 0, 1, BASE, LEGS);
                        let divisor_tensor = create_constant_tensor(self.divisor, 1);
                        let half_divisor_int = felt_to_integer_rep(self.divisor) / 2;
                        let half_divisor_tensor: ValTensor<F> =
                            create_constant_tensor(integer_rep_to_felt(half_divisor_int), 1);
                        let (base, legs) = (region.base(), region.legs());

                        let (q_digits, claimed_q) = layouts::decompose(
                            &config,
                            &mut region,
                            &[&self.claimed_q],
                            &base,
                            &legs,
                            true,
                        )?;

                        let product = layouts::pairwise(
                            &config,
                            &mut region,
                            &[&claimed_q, &divisor_tensor],
                            BaseOp::Mult,
                        )?;
                        let reconstructed = layouts::pairwise(
                            &config,
                            &mut region,
                            &[&product, &self.claimed_r],
                            BaseOp::Add,
                        )?;
                        layouts::enforce_equality(
                            &config,
                            &mut region,
                            &[&self.input, &reconstructed],
                        )?;

                        let shifted_r = layouts::pairwise(
                            &config,
                            &mut region,
                            &[&self.claimed_r, &half_divisor_tensor],
                            BaseOp::Add,
                        )?;
                        let shared_range: crate::circuit::table::Range = (0, BASE as i128);
                        let divisor_int = felt_to_integer_rep(self.divisor);

                        if divisor_int <= BASE as i128 {
                            layouts::range_check(
                                &config,
                                &mut region,
                                &[&shifted_r],
                                &shared_range,
                            )?;

                            if divisor_int != BASE as i128 {
                                let gap = layouts::pairwise(
                                    &config,
                                    &mut region,
                                    &[&divisor_tensor, &shifted_r],
                                    BaseOp::Sub,
                                )?;
                                layouts::range_check(&config, &mut region, &[&gap], &shared_range)?;
                            }
                        } else {
                            let n = layouts::limbs_for_bound(divisor_int, BASE as i128)?;
                            layouts::range_check_unsigned_decomp(
                                &config,
                                &mut region,
                                &shifted_r,
                                BASE,
                                n,
                            )?;
                            let gap = layouts::pairwise(
                                &config,
                                &mut region,
                                &[&divisor_tensor, &shifted_r],
                                BaseOp::Sub,
                            )?;
                            layouts::range_check_unsigned_decomp(
                                &config,
                                &mut region,
                                &gap,
                                BASE,
                                n,
                            )?;
                        }

                        let mut sign_q = q_digits;
                        sign_q.get_every_n(region.legs() + 1)?;
                        sign_q.reshape(self.claimed_q.dims())?;

                        let zero_tensor = create_zero_tensor(1);
                        let at_bottom =
                            layouts::equals(&config, &mut region, &[&shifted_r, &zero_tensor])?;
                        let at_top =
                            layouts::equals(&config, &mut region, &[&shifted_r, &divisor_tensor])?;

                        let one_tensor = create_constant_tensor(F::ONE, 1);
                        let zero_check_tensor = create_constant_tensor(F::ZERO, sign_q.len());

                        let one_minus_sign_q = layouts::pairwise(
                            &config,
                            &mut region,
                            &[&one_tensor, &sign_q],
                            BaseOp::Sub,
                        )?;
                        let bottom_violation = layouts::pairwise(
                            &config,
                            &mut region,
                            &[&at_bottom, &one_minus_sign_q],
                            BaseOp::Mult,
                        )?;
                        layouts::enforce_equality(
                            &config,
                            &mut region,
                            &[&bottom_violation, &zero_check_tensor],
                        )?;

                        let one_plus_sign_q = layouts::pairwise(
                            &config,
                            &mut region,
                            &[&one_tensor, &sign_q],
                            BaseOp::Add,
                        )?;
                        let top_violation = layouts::pairwise(
                            &config,
                            &mut region,
                            &[&at_top, &one_plus_sign_q],
                            BaseOp::Mult,
                        )?;
                        layouts::enforce_equality(
                            &config,
                            &mut region,
                            &[&top_violation, &zero_check_tensor],
                        )?;

                        Ok(())
                    };
                    body().map_err(|_| Error::Synthesis)
                },
            )?;

            Ok(())
        }
    }

    fn bad_div_round_nearest_accepted(
        divisor_int: i128,
        input_int: i128,
        bad_q_int: i128,
        bad_r_int: i128,
    ) -> bool {
        let circuit = BadDivRoundNearestCircuit::<F> {
            input: single_value(integer_rep_to_felt(input_int)),
            claimed_q: single_value(integer_rep_to_felt(bad_q_int)),
            claimed_r: single_value(integer_rep_to_felt(bad_r_int)),
            divisor: integer_rep_to_felt(divisor_int),
        };

        match MockProver::run(K, &circuit, vec![]) {
            Ok(prover) => prover.verify().is_ok(),
            Err(_) => false,
        }
    }

    #[test]
    fn negative_a_wrong_quotient_rejected_by_constraint_a() {
        assert!(bad_div_round_nearest_accepted(16, 53, 3, 5));

        assert!(!bad_div_round_nearest_accepted(16, 53, 6, 5));
        assert!(!bad_div_round_nearest_accepted(16, 53, 0, 5));
        assert!(!bad_div_round_nearest_accepted(16, -53, 3, -8));

        for &d in &[8192i128, 16384] {
            let half = d / 2;
            assert!(bad_div_round_nearest_accepted(d, 3 * d + 7, 3, 7));
            assert!(!bad_div_round_nearest_accepted(d, 3 * d + 7, 5, 7));
            let _ = half;
        }
    }

    #[test]
    fn negative_b_out_of_window_remainder_rejected_by_constraint_b() {
        for &d in &[16i128, 8192, 16384] {
            assert!(!bad_div_round_nearest_accepted(d, d, 0, d));
            assert!(!bad_div_round_nearest_accepted(d, -(d + 1), 0, -(d + 1)));
        }
    }

    #[test]
    fn negative_b_wide_divisor_out_of_window_remainder_rejected() {
        for &d in WIDE_DIVISORS.iter() {
            let half = d / 2;

            assert!(bad_div_round_nearest_accepted(d, 3 * d + 7, 3, 7));
            assert!(bad_div_round_nearest_accepted(d, 3 * d - half, 3, -half));
            assert!(bad_div_round_nearest_accepted(d, -3 * d + half, -3, half));

            assert!(!bad_div_round_nearest_accepted(d, half + 1, 0, half + 1));

            assert!(!bad_div_round_nearest_accepted(d, -half - 1, 0, -half - 1));

            assert!(!bad_div_round_nearest_accepted(d, d, 0, d));
        }
    }

    #[test]
    fn negative_gap_soundness_shared_table() {
        for &d in &[16i128, 8192] {
            let half = d / 2;

            assert!(bad_div_round_nearest_accepted(d, 2 * d, 2, 0));

            assert!(!bad_div_round_nearest_accepted(
                d,
                2 * d + half + 1,
                2,
                half + 1
            ));
        }
    }

    #[test]
    fn negative_d_tie_toward_zero_rejected_with_honest_sign() {
        for &d in &[16i128, 8192, 16384] {
            let half = d / 2;
            assert!(bad_div_round_nearest_accepted(d, half, 1, -half));
            assert!(!bad_div_round_nearest_accepted(d, half, 0, half));

            assert!(bad_div_round_nearest_accepted(d, -half, -1, half));
            assert!(!bad_div_round_nearest_accepted(d, -half, 0, -half));
        }
    }

    #[derive(Clone)]
    struct ForgedSignCircuit<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> {
        input: F,
        q_val: F,
        r_val: F,
        forged_sign: F,
        divisor: F,
        apply_zero_sign_matters: bool,
    }

    impl Circuit<F> for ForgedSignCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            configure_shared(cs)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.layout_range_checks(&mut layouter).unwrap();
            layouter.assign_region(
                || "",
                |region| {
                    let body = || -> Result<(), crate::circuit::CircuitError> {
                        let mut region = RegionCtx::new(region, 0, 1, BASE, LEGS);

                        let input_t = single_value(self.input);
                        let q_t = single_value(self.q_val);
                        let r_t = single_value(self.r_val);
                        let sign_t = single_value(self.forged_sign);

                        let divisor_tensor = create_constant_tensor(self.divisor, 1);
                        let half_divisor_int = felt_to_integer_rep(self.divisor) / 2;
                        let half_divisor_tensor: ValTensor<F> =
                            create_constant_tensor(integer_rep_to_felt(half_divisor_int), 1);

                        let sign_checked =
                            layouts::range_check(&config, &mut region, &[&sign_t], &(-1, 1))?;

                        if self.apply_zero_sign_matters {
                            let is_zero = layouts::equals_zero(&config, &mut region, &[&q_t])?;
                            let sign_is_zero = layouts::pairwise(
                                &config,
                                &mut region,
                                &[&sign_checked, &is_zero],
                                BaseOp::Mult,
                            )?;
                            layouts::enforce_equality(
                                &config,
                                &mut region,
                                &[&sign_is_zero, &create_zero_tensor(1)],
                            )?;
                        }

                        let product = layouts::pairwise(
                            &config,
                            &mut region,
                            &[&q_t, &divisor_tensor],
                            BaseOp::Mult,
                        )?;
                        let reconstructed = layouts::pairwise(
                            &config,
                            &mut region,
                            &[&product, &r_t],
                            BaseOp::Add,
                        )?;
                        layouts::enforce_equality(
                            &config,
                            &mut region,
                            &[&input_t, &reconstructed],
                        )?;

                        let shifted_r = layouts::pairwise(
                            &config,
                            &mut region,
                            &[&r_t, &half_divisor_tensor],
                            BaseOp::Add,
                        )?;
                        let shared_range: crate::circuit::table::Range = (0, BASE as i128);
                        layouts::range_check(&config, &mut region, &[&shifted_r], &shared_range)?;

                        let divisor_int = felt_to_integer_rep(self.divisor);
                        if divisor_int != BASE as i128 {
                            let gap = layouts::pairwise(
                                &config,
                                &mut region,
                                &[&divisor_tensor, &shifted_r],
                                BaseOp::Sub,
                            )?;
                            layouts::range_check(&config, &mut region, &[&gap], &shared_range)?;
                        }

                        let zero_tensor = create_zero_tensor(1);
                        let at_bottom =
                            layouts::equals(&config, &mut region, &[&shifted_r, &zero_tensor])?;
                        let at_top =
                            layouts::equals(&config, &mut region, &[&shifted_r, &divisor_tensor])?;

                        let one_tensor = create_constant_tensor(F::ONE, 1);
                        let zero_check_tensor = create_constant_tensor(F::ZERO, 1);

                        let one_minus_sign = layouts::pairwise(
                            &config,
                            &mut region,
                            &[&one_tensor, &sign_checked],
                            BaseOp::Sub,
                        )?;
                        let bottom_violation = layouts::pairwise(
                            &config,
                            &mut region,
                            &[&at_bottom, &one_minus_sign],
                            BaseOp::Mult,
                        )?;
                        layouts::enforce_equality(
                            &config,
                            &mut region,
                            &[&bottom_violation, &zero_check_tensor],
                        )?;

                        let one_plus_sign = layouts::pairwise(
                            &config,
                            &mut region,
                            &[&one_tensor, &sign_checked],
                            BaseOp::Add,
                        )?;
                        let top_violation = layouts::pairwise(
                            &config,
                            &mut region,
                            &[&at_top, &one_plus_sign],
                            BaseOp::Mult,
                        )?;
                        layouts::enforce_equality(
                            &config,
                            &mut region,
                            &[&top_violation, &zero_check_tensor],
                        )?;

                        Ok(())
                    };
                    body().map_err(|_| Error::Synthesis)
                },
            )?;

            Ok(())
        }
    }

    fn forged_sign_accepted(
        divisor_int: i128,
        input_int: i128,
        q_int: i128,
        r_int: i128,
        forged_sign_int: i128,
        apply_zero_sign_matters: bool,
    ) -> bool {
        let circuit = ForgedSignCircuit::<F> {
            input: integer_rep_to_felt(input_int),
            q_val: integer_rep_to_felt(q_int),
            r_val: integer_rep_to_felt(r_int),
            forged_sign: integer_rep_to_felt(forged_sign_int),
            divisor: integer_rep_to_felt(divisor_int),
            apply_zero_sign_matters,
        };
        match MockProver::run(K, &circuit, vec![]) {
            Ok(prover) => prover.verify().is_ok(),
            Err(_) => false,
        }
    }

    #[test]
    fn negative_c_forged_sign_at_zero_quotient() {
        for &d in &[16i128, 16384] {
            let half = d / 2;

            assert!(!forged_sign_accepted(d, half, 0, half, -1, true));
            assert!(forged_sign_accepted(d, half, 0, half, -1, false));

            assert!(!forged_sign_accepted(d, -half, 0, -half, 1, true));
            assert!(forged_sign_accepted(d, -half, 0, -half, 1, false));

            assert!(!forged_sign_accepted(d, half, 0, half, 1, true));
            assert!(!forged_sign_accepted(d, half, 0, half, 1, false));
            assert!(!forged_sign_accepted(d, -half, 0, -half, -1, true));
            assert!(!forged_sign_accepted(d, -half, 0, -half, -1, false));
        }
    }

    #[test]
    fn negative_d_honest_zero_sign_isolated_gadget() {
        for &d in &[16i128, 16384] {
            let half = d / 2;
            assert!(!forged_sign_accepted(d, half, 0, half, 0, true));
            assert!(!forged_sign_accepted(d, half, 0, half, 0, false));
            assert!(!forged_sign_accepted(d, -half, 0, -half, 0, true));
            assert!(!forged_sign_accepted(d, -half, 0, -half, 0, false));
        }
    }

    #[derive(Clone)]
    struct DivIdentityCircuit<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> {
        input: ValTensor<F>,
    }

    impl Circuit<F> for DivIdentityCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            configure_shared(cs)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.layout_range_checks(&mut layouter).unwrap();
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, BASE, LEGS);
                        let out_old =
                            layouts::div(&config, &mut region, &[&self.input], F::ONE).unwrap();
                        let out_new = layouts::div_round_nearest(
                            &config,
                            &mut region,
                            &[&self.input],
                            F::ONE,
                        )
                        .unwrap();
                        layouts::enforce_equality(&config, &mut region, &[&self.input, &out_old])
                            .unwrap();
                        layouts::enforce_equality(&config, &mut region, &[&self.input, &out_new])
                            .unwrap();
                        Ok(())
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    #[test]
    fn divisor_one_short_circuit_matches_old_and_new() {
        let inputs = [0i128, 1, -1, 12345, -12345, 65535, -65535];
        let input: Tensor<Value<F>> = Tensor::new(
            Some(
                &inputs
                    .iter()
                    .map(|&x| Value::known(integer_rep_to_felt(x)))
                    .collect::<Vec<_>>(),
            ),
            &[inputs.len()],
        )
        .unwrap();

        let circuit = DivIdentityCircuit::<F> {
            input: ValTensor::from(input),
        };

        let prover = MockProver::run(K, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    fn range_check_cols_and_col_size(range: (i128, i128), k: usize) -> (usize, usize) {
        let mut cs = ConstraintSystem::<F>::default();
        let advices = (0..3)
            .map(|_| VarTensor::new_advice(&mut cs, k, 1, 1024))
            .collect::<Vec<_>>();
        let mut config = BaseConfig::configure(
            &mut cs,
            &[advices[0].clone(), advices[1].clone()],
            &advices[2],
            CheckMode::SAFE,
        );
        config
            .configure_range_check(&mut cs, &advices[0], &advices[1], range, k)
            .unwrap();
        let rc = config.range_checks.ranges.get(&range).unwrap();
        (rc.inputs.len(), rc.col_size)
    }

    #[test]
    fn table_width_cost_report() {
        println!("\n--- shared range-check table cost: window (0,16384) ---");
        let mut results = vec![];
        for k in [13usize, 14, 15, 17, 20] {
            let (cols, col_size) = range_check_cols_and_col_size((0, 16384), k);
            println!(
                "K={:>2}  col_size(rows usable)={:<10}  columns_required={}",
                k, col_size, cols
            );
            results.push((k, cols, col_size));
        }

        println!("--- old scheme's fixed-size tables (independent of divisor) ---");
        for &(label, range) in &[("sign [-1,1]", (-1i128, 1i128)), ("digit [0,15]", (0, 15))] {
            let (cols, col_size) = range_check_cols_and_col_size(range, 15);
            println!(
                "{:<14} K=15  col_size={:<10} columns_required={}",
                label, col_size, cols
            );
        }

        let cols_at = |k: usize| results.iter().find(|&&(kk, _, _)| kk == k).unwrap().1;
        assert_eq!(cols_at(13), 3);
        assert_eq!(cols_at(14), 2);
        assert_eq!(cols_at(15), 1);
        assert_eq!(cols_at(17), 1);
        assert_eq!(cols_at(20), 1);

        assert_eq!(range_check_cols_and_col_size((-1, 1), 13).0, 1);
        assert_eq!(range_check_cols_and_col_size((0, 15), 13).0, 1);
    }

    fn layout_err(divisor_int: i128, x: i128) -> crate::circuit::CircuitError {
        use crate::circuit::ops::region::RegionSettings;
        let dummy_config = BaseConfig::<F>::dummy(12, 2);
        let mut dummy_region = RegionCtx::new_dummy(0, 2, RegionSettings::all_true(BASE, LEGS));
        let input = single_value::<F>(integer_rep_to_felt(x));
        layouts::div_round_nearest::<F>(
            &dummy_config,
            &mut dummy_region,
            &[&input],
            integer_rep_to_felt(divisor_int),
        )
        .unwrap_err()
    }

    #[test]
    fn odd_divisor_is_refused_not_silently_miscomputed() {
        for d in [3i128, 5, 7, 9, 15] {
            match layout_err(d, -1) {
                crate::circuit::CircuitError::InvalidDivisor(got) => assert_eq!(got, d),
                other => panic!("divisor {d} should be refused, got {other:?}"),
            }
        }
    }

    #[test]
    fn non_positive_divisor_is_refused() {
        for d in [0i128, -2, -16] {
            match layout_err(d, 5) {
                crate::circuit::CircuitError::InvalidDivisor(got) => assert_eq!(got, d),
                other => panic!("divisor {d} should be refused, got {other:?}"),
            }
        }
    }

    #[test]
    fn even_non_power_of_two_divisor_still_computes_round_to_nearest() {
        use crate::circuit::ops::region::RegionSettings;
        let dummy_config = BaseConfig::<F>::dummy(12, 2);
        for (x, expected) in [(0i128, 0i128), (7, 1), (9, 2), (-7, -1), (-9, -2), (18, 3)] {
            let mut dummy_region = RegionCtx::new_dummy(0, 2, RegionSettings::all_true(BASE, LEGS));
            let input = single_value::<F>(integer_rep_to_felt(x));
            let out = layouts::div_round_nearest::<F>(
                &dummy_config,
                &mut dummy_region,
                &[&input],
                integer_rep_to_felt(6),
            )
            .unwrap();
            assert_eq!(
                out.int_evals().unwrap()[0],
                expected,
                "6 divides {x} to {expected}"
            );
        }
    }

    #[test]
    fn dispatch_admits_only_positive_powers_of_two() {
        use crate::circuit::ops::hybrid::div_round_nearest_applies as applies;
        for d in [2i128, 4, 16, 128, 8192, 16384, 1 << 21] {
            assert!(applies(d), "{d} is proven");
        }
        for d in [3i128, 5, 6, 10, 12, 0, -2, -16] {
            assert!(!applies(d), "{d} is outside the proven range");
        }
    }
}

#[cfg(test)]
#[cfg(all(
    feature = "ezkl",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
mod lookup_ultra_overflow {
    use super::*;
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{Circuit, ConstraintSystem, Error},
        poly::kzg::{
            commitment::KZGCommitmentScheme,
            multiopen::{ProverSHPLONK, VerifierSHPLONK},
            strategy::SingleStrategy,
        },
    };
    use snark_verifier::system::halo2::transcript::evm::EvmTranscript;

    #[derive(Clone)]
    struct SigmoidCircuit<F: PrimeField + TensorType + PartialOrd> {
        pub input: ValTensor<F>,
    }

    impl Circuit<F> for SigmoidCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let advices = (0..3)
                .map(|_| VarTensor::new_advice(cs, 4, 1, 3))
                .collect::<Vec<_>>();

            let nl = LookupOp::Sigmoid { scale: 1.0.into() };

            let mut config = BaseConfig::default();

            config
                .configure_lookup(
                    cs,
                    &advices[0],
                    &advices[1],
                    &advices[2],
                    (-1024, 1024),
                    4,
                    &nl,
                )
                .unwrap();
            config
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>, // layouter is our 'write buffer' for the circuit
        ) -> Result<(), Error> {
            config.layout_tables(&mut layouter).unwrap();
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &[&self.input],
                                Box::new(LookupOp::Sigmoid { scale: 1.0.into() }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn sigmoidcircuit() {
        // get some logs fam
        crate::logger::init_logger();
        // parameters
        let a = Tensor::from((0..4).map(|i| Value::known(F::from(i + 1))));

        let circuit = SigmoidCircuit::<F> {
            input: ValTensor::from(a),
        };

        let params = crate::pfsys::srs::gen_srs::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<_>,
        >(4_u32);

        let pk = crate::pfsys::create_keys::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            SigmoidCircuit<F>,
        >(&circuit, &params, true)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit::<
            KZGCommitmentScheme<_>,
            _,
            ProverSHPLONK<_>,
            VerifierSHPLONK<_>,
            SingleStrategy<_>,
            _,
            EvmTranscript<_, _, _, _>,
            EvmTranscript<_, _, _, _>,
        >(
            circuit.clone(),
            vec![],
            &params,
            &pk,
            // use safe mode to verify that the proof is correct
            CheckMode::SAFE,
            None,
            None,
        );

        assert!(prover.is_ok());
    }
}
