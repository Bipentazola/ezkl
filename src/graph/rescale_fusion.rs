use std::collections::{BTreeMap, HashMap};

use log::debug;

use crate::circuit::ops::poly::PolyOp;
use crate::circuit::ops::Op;
use crate::graph::errors::GraphError;
use crate::graph::model::NodeType;
use crate::graph::node::{RebaseScale, SupportedOp};
use crate::graph::vars::VarScales;
use crate::RunArgs;

const ENABLE_VAR: &str = "EZKL_RESCALE_FUSION";

const MAX_DIVISOR_BITS_VAR: &str = "EZKL_RESCALE_FUSION_MAX_DIVISOR_BITS";

const DEFAULT_MAX_DIVISOR_BITS: i32 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusionConfig {
    pub max_divisor_bits: i32,
    pub wide_divisors: bool,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            max_divisor_bits: DEFAULT_MAX_DIVISOR_BITS,
            wide_divisors: true,
        }
    }
}

impl FusionConfig {
    pub fn from_env() -> Option<Self> {
        if std::env::var(ENABLE_VAR).as_deref() != Ok("1") {
            return None;
        }
        let max_divisor_bits = std::env::var(MAX_DIVISOR_BITS_VAR)
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(DEFAULT_MAX_DIVISOR_BITS);
        Some(Self {
            max_divisor_bits,
            wide_divisors: *crate::circuit::ops::hybrid::EZKL_DIV_ROUND_NEAREST,
        })
    }
}

pub fn narrow_path_max_divisor_bits(run_args: &RunArgs) -> i32 {
    let ceiling = (run_args.decomp_base as u128)
        .checked_pow(run_args.decomp_legs as u32)
        .unwrap_or(u128::MAX);
    match ceiling.checked_sub(1) {
        Some(0) | None => 0,
        Some(max_representable) => (127 - max_representable.leading_zeros()) as i32,
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct FusionStats {
    pub fused: usize,
    pub rejected_over_budget: usize,
    pub max_divisor_bits_used: i32,
}

fn is_pure_rescale_inner(op: &SupportedOp) -> bool {
    matches!(op, SupportedOp::Linear(PolyOp::Mult))
}

fn can_absorb_delayed_scale(op: &SupportedOp) -> bool {
    matches!(
        op,
        SupportedOp::Linear(PolyOp::Mult) | SupportedOp::Linear(PolyOp::Einsum { .. })
    )
}

fn consumer_map(nodes: &BTreeMap<usize, NodeType>) -> HashMap<usize, Vec<usize>> {
    let mut consumers: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, node) in nodes.iter() {
        for (producer, _slot) in node.inputs() {
            consumers.entry(producer).or_default().push(*idx);
        }
    }
    consumers
}

fn current_in_scales(
    nodes: &BTreeMap<usize, NodeType>,
    idx: usize,
) -> Result<Vec<crate::Scale>, GraphError> {
    nodes
        .get(&idx)
        .ok_or(GraphError::MissingNode(idx))?
        .inputs()
        .iter()
        .map(|(producer, slot)| {
            let producer_node = nodes
                .get(producer)
                .ok_or(GraphError::MissingNode(*producer))?;
            producer_node
                .out_scales()
                .get(*slot)
                .copied()
                .ok_or(GraphError::MissingNode(*producer))
        })
        .collect()
}

fn rebuild_consumer(
    inner: SupportedOp,
    in_scales: Vec<crate::Scale>,
    scales: &VarScales,
    run_args: &RunArgs,
) -> Result<Option<(SupportedOp, crate::Scale, i32)>, GraphError> {
    let op_out_scale = inner.out_scale(in_scales.clone())?;
    let rebase_scale = scales.get_rebase_scale();
    let rebased = RebaseScale::rebase(
        inner,
        rebase_scale,
        op_out_scale,
        run_args.scale_rebase_multiplier,
    );

    let divisor_bits = match &rebased {
        SupportedOp::RebaseScale(rb) => rb.original_scale - rb.target_scale,
        _ => return Ok(None),
    };

    let out_scale = rebased.out_scale(in_scales)?;
    Ok(Some((rebased, out_scale, divisor_bits)))
}

pub fn fuse_rescales(
    nodes: &mut BTreeMap<usize, NodeType>,
    outputs: &[(usize, usize)],
    scales: &VarScales,
    run_args: &RunArgs,
    config: &FusionConfig,
) -> Result<FusionStats, GraphError> {
    let mut stats = FusionStats::default();

    let narrow_ceiling = narrow_path_max_divisor_bits(run_args);
    let budget = if config.wide_divisors {
        config.max_divisor_bits
    } else {
        config.max_divisor_bits.min(narrow_ceiling)
    };

    let consumers = consumer_map(nodes);
    let output_nodes: std::collections::HashSet<usize> =
        outputs.iter().map(|(idx, _)| *idx).collect();

    let candidate_indices: Vec<usize> = nodes.keys().copied().collect();
    debug!(
        "rescale fusion: enabled, budget 2^{}{}, scanning {} nodes",
        budget,
        if config.wide_divisors {
            String::new()
        } else {
            format!(
                " (tightened from 2^{} -- EZKL_DIV_ROUND_NEAREST is off, so a fused \
                 divisor must stay representable by base {} and {} legs)",
                config.max_divisor_bits, run_args.decomp_base, run_args.decomp_legs
            )
        },
        candidate_indices.len()
    );

    for idx in candidate_indices {
        if output_nodes.contains(&idx) {
            continue;
        }

        let (inner, delayed_scale) = match nodes.get(&idx) {
            Some(NodeType::Node(n)) => match &n.opkind {
                SupportedOp::RebaseScale(rb) if is_pure_rescale_inner(&rb.inner) => {
                    (*rb.inner.clone(), rb.original_scale)
                }
                _ => continue,
            },
            _ => continue,
        };

        let my_consumers = match consumers.get(&idx) {
            Some(c) if !c.is_empty() => c.clone(),
            _ => continue,
        };

        let consumers_eligible = my_consumers.iter().all(|c| match nodes.get(c) {
            Some(NodeType::Node(n)) => match &n.opkind {
                SupportedOp::RebaseScale(rb) => can_absorb_delayed_scale(&rb.inner),
                _ => false,
            },
            _ => false,
        });
        if !consumers_eligible {
            continue;
        }

        let mut rebuilt: Vec<(usize, SupportedOp, crate::Scale)> = Vec::new();
        let mut over_budget = false;
        let mut escaped = false;
        let mut widest = 0;

        for c in &my_consumers {
            let consumer_inner = match nodes.get(c) {
                Some(NodeType::Node(n)) => match &n.opkind {
                    SupportedOp::RebaseScale(rb) => *rb.inner.clone(),
                    _ => unreachable!("eligibility checked above"),
                },
                _ => unreachable!("eligibility checked above"),
            };

            let mut in_scales = current_in_scales(nodes, *c)?;
            for ((producer, _), s) in nodes
                .get(c)
                .ok_or(GraphError::MissingNode(*c))?
                .inputs()
                .iter()
                .zip(in_scales.iter_mut())
            {
                if *producer == idx {
                    *s = delayed_scale;
                }
            }

            match rebuild_consumer(consumer_inner, in_scales, scales, run_args)? {
                Some((op, out_scale, divisor_bits)) => {
                    if divisor_bits > budget {
                        over_budget = true;
                        break;
                    }
                    widest = widest.max(divisor_bits);
                    rebuilt.push((*c, op, out_scale));
                }
                None => {
                    escaped = true;
                    break;
                }
            }
        }

        if over_budget {
            stats.rejected_over_budget += 1;
            continue;
        }
        if escaped {
            continue;
        }

        if let Some(node) = nodes.get_mut(&idx) {
            node.replace_opkind(inner);
            node.bump_scale(delayed_scale);
        }
        for (c, op, out_scale) in rebuilt {
            if let Some(node) = nodes.get_mut(&c) {
                node.replace_opkind(op);
                node.bump_scale(out_scale);
            }
        }
        stats.fused += 1;
        stats.max_divisor_bits_used = stats.max_divisor_bits_used.max(widest);
    }

    if stats.fused > 0 || stats.rejected_over_budget > 0 {
        debug!(
            "rescale fusion: delayed {} rescale(s) into their consumers, \
             rejected {} over the {}-bit divisor budget{}, widest fused divisor 2^{}",
            stats.fused,
            stats.rejected_over_budget,
            budget,
            if !config.wide_divisors && stats.rejected_over_budget > 0 {
                " (the decomposition's ceiling -- set EZKL_DIV_ROUND_NEAREST=1 to fuse further)"
            } else {
                ""
            },
            stats.max_divisor_bits_used
        );
    }

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::ops::lookup::LookupOp;
    use crate::graph::node::Node;

    const SCALE: crate::Scale = 7;

    fn run_args() -> RunArgs {
        let mut args = RunArgs::default();
        args.input_scale = SCALE;
        args.param_scale = SCALE;
        args.scale_rebase_multiplier = 1;
        args
    }

    fn rebased(inner: SupportedOp, in_scales: &[crate::Scale]) -> (SupportedOp, crate::Scale) {
        let op_out_scale = inner.out_scale(in_scales.to_vec()).unwrap();
        let op = RebaseScale::rebase(inner, SCALE, op_out_scale, 1);
        let out_scale = op.out_scale(in_scales.to_vec()).unwrap();
        (op, out_scale)
    }

    fn node(
        idx: usize,
        opkind: SupportedOp,
        out_scale: crate::Scale,
        inputs: Vec<usize>,
    ) -> NodeType {
        NodeType::Node(Node {
            opkind,
            out_scale,
            inputs: inputs.into_iter().map(|i| (i, 0)).collect(),
            out_dims: vec![1],
            idx,
            num_uses: 1,
        })
    }

    fn input_node(idx: usize) -> NodeType {
        node(
            idx,
            SupportedOp::Input(crate::circuit::Input {
                scale: SCALE,
                datum_type: crate::circuit::InputType::F32,
                decomp: false,
            }),
            SCALE,
            vec![],
        )
    }

    fn mult_chain(
        len: usize,
        tail: Option<SupportedOp>,
    ) -> (BTreeMap<usize, NodeType>, Vec<usize>) {
        let mut nodes = BTreeMap::new();
        nodes.insert(0, input_node(0));
        nodes.insert(1, input_node(1));

        let mut prev = 0usize;
        let mut chain = vec![];
        for i in 0..len {
            let idx = 2 + i;
            let prev_scale = nodes[&prev].out_scales()[0];
            let (op, out_scale) = rebased(SupportedOp::Linear(PolyOp::Mult), &[prev_scale, SCALE]);
            nodes.insert(idx, node(idx, op, out_scale, vec![prev, 1]));
            chain.push(idx);
            prev = idx;
        }

        if let Some(tail_op) = tail {
            let idx = 2 + len;
            let prev_scale = nodes[&prev].out_scales()[0];
            let out_scale = tail_op.out_scale(vec![prev_scale, prev_scale]).unwrap();
            nodes.insert(idx, node(idx, tail_op, out_scale, vec![prev, prev]));
        }
        (nodes, chain)
    }

    fn fuse(
        nodes: &mut BTreeMap<usize, NodeType>,
        outputs: &[(usize, usize)],
        max_divisor_bits: i32,
    ) -> FusionStats {
        fuse_with(nodes, outputs, max_divisor_bits, true)
    }

    fn fuse_with(
        nodes: &mut BTreeMap<usize, NodeType>,
        outputs: &[(usize, usize)],
        max_divisor_bits: i32,
        wide_divisors: bool,
    ) -> FusionStats {
        let args = run_args();
        let scales = VarScales::from_args(&args);
        fuse_rescales(
            nodes,
            outputs,
            &scales,
            &args,
            &FusionConfig {
                max_divisor_bits,
                wide_divisors,
            },
        )
        .unwrap()
    }

    fn divisor_bits(nodes: &BTreeMap<usize, NodeType>, idx: usize) -> Option<i32> {
        match &nodes[&idx] {
            NodeType::Node(n) => match &n.opkind {
                SupportedOp::RebaseScale(rb) => Some(rb.original_scale - rb.target_scale),
                _ => None,
            },
            _ => None,
        }
    }

    #[test]
    fn collapses_a_chain_into_one_rescale() {
        let (mut nodes, chain) = mult_chain(3, Some(SupportedOp::Linear(PolyOp::Add)));
        for &idx in &chain {
            assert_eq!(
                divisor_bits(&nodes, idx),
                Some(SCALE),
                "setup: every link rescales by 2^7"
            );
        }

        let stats = fuse(&mut nodes, &[], 64);

        assert_eq!(stats.fused, 2, "the first two links should be delayed");
        assert_eq!(stats.rejected_over_budget, 0);
        assert_eq!(divisor_bits(&nodes, chain[0]), None);
        assert_eq!(divisor_bits(&nodes, chain[1]), None);
        assert_eq!(divisor_bits(&nodes, chain[2]), Some(3 * SCALE));
        assert_eq!(stats.max_divisor_bits_used, 3 * SCALE);
        assert_eq!(nodes[&chain[2]].out_scales()[0], SCALE);
    }

    #[test]
    fn budget_caps_the_fused_divisor() {
        let (mut nodes, chain) = mult_chain(3, Some(SupportedOp::Linear(PolyOp::Add)));
        let stats = fuse(&mut nodes, &[], 2 * SCALE);

        assert_eq!(stats.fused, 1);
        assert_eq!(stats.rejected_over_budget, 1);
        assert_eq!(divisor_bits(&nodes, chain[0]), None);
        assert_eq!(divisor_bits(&nodes, chain[1]), Some(2 * SCALE));
        assert_eq!(divisor_bits(&nodes, chain[2]), Some(SCALE));
    }

    #[test]
    fn refuses_when_the_consumer_does_not_rescale() {
        let (mut nodes, chain) = mult_chain(1, None);
        let sink = 3usize;
        let prev_scale = nodes[&chain[0]].out_scales()[0];
        let tail = SupportedOp::Nonlinear(LookupOp::Sigmoid {
            scale: crate::circuit::utils::F32(128.0),
        });
        let out_scale = tail.out_scale(vec![prev_scale]).unwrap();
        nodes.insert(sink, node(sink, tail, out_scale, vec![chain[0]]));

        let stats = fuse(&mut nodes, &[], 64);
        assert_eq!(stats.fused, 0);
        assert_eq!(divisor_bits(&nodes, chain[0]), Some(SCALE));
    }

    #[test]
    fn refuses_a_consumer_that_requires_homogenous_input_scales() {
        let (mut nodes, chain) = mult_chain(1, None);
        let sink = 3usize;
        let prev_scale = nodes[&chain[0]].out_scales()[0];
        let (op, out_scale) = rebased(SupportedOp::Linear(PolyOp::Add), &[prev_scale, prev_scale]);
        nodes.insert(sink, node(sink, op, out_scale, vec![chain[0], chain[0]]));

        let stats = fuse(&mut nodes, &[], 64);
        assert_eq!(stats.fused, 0);
        assert_eq!(divisor_bits(&nodes, chain[0]), Some(SCALE));
    }

    #[test]
    fn never_delays_a_graph_output() {
        let (mut nodes, chain) = mult_chain(2, None);
        let stats = fuse(&mut nodes, &[(chain[0], 0)], 64);
        assert_eq!(
            stats.fused, 0,
            "an output node's scale is part of the model's contract"
        );
        assert_eq!(divisor_bits(&nodes, chain[0]), Some(SCALE));
    }

    #[test]
    fn only_pure_rescale_nodes_are_delayed() {
        let mut nodes = BTreeMap::new();
        nodes.insert(0, input_node(0));
        nodes.insert(1, input_node(1));
        let (einsum, es) = rebased(
            SupportedOp::Linear(PolyOp::Einsum {
                equation: "ij,jk->ik".to_string(),
            }),
            &[SCALE, SCALE],
        );
        nodes.insert(2, node(2, einsum, es, vec![0, 1]));
        let (mult, ms) = rebased(SupportedOp::Linear(PolyOp::Mult), &[es, SCALE]);
        nodes.insert(3, node(3, mult, ms, vec![2, 1]));

        let stats = fuse(&mut nodes, &[], 64);
        assert_eq!(stats.fused, 0);
        assert_eq!(
            divisor_bits(&nodes, 2),
            Some(SCALE),
            "the einsum keeps its own rescale"
        );
    }

    #[test]
    fn fans_out_to_every_consumer_or_none() {
        let mut nodes = BTreeMap::new();
        nodes.insert(0, input_node(0));
        nodes.insert(1, input_node(1));
        let (mult, ms) = rebased(SupportedOp::Linear(PolyOp::Mult), &[SCALE, SCALE]);
        nodes.insert(2, node(2, mult, ms, vec![0, 1]));
        for c in 3..6 {
            let (einsum, es) = rebased(
                SupportedOp::Linear(PolyOp::Einsum {
                    equation: "ij,jk->ik".to_string(),
                }),
                &[ms, SCALE],
            );
            nodes.insert(c, node(c, einsum, es, vec![2, 1]));
        }

        let stats = fuse(&mut nodes, &[], 64);
        assert_eq!(stats.fused, 1);
        assert_eq!(divisor_bits(&nodes, 2), None);
        for c in 3..6 {
            assert_eq!(divisor_bits(&nodes, c), Some(2 * SCALE), "consumer {c}");
        }
    }

    #[test]
    fn disabled_by_default() {
        assert!(
            FusionConfig::from_env().is_none() || std::env::var(ENABLE_VAR).as_deref() == Ok("1"),
            "the pass must be opt-in"
        );
        assert_eq!(
            FusionConfig::default().max_divisor_bits,
            DEFAULT_MAX_DIVISOR_BITS
        );
    }

    #[test]
    fn the_narrow_ceiling_is_the_decomposition_ceiling() {
        assert_eq!(narrow_path_max_divisor_bits(&run_args()), 27);

        let mut args = run_args();
        args.decomp_base = 1024;
        args.decomp_legs = 3;
        assert_eq!(narrow_path_max_divisor_bits(&args), 29);

        args.decomp_base = usize::MAX;
        args.decomp_legs = 64;
        assert_eq!(narrow_path_max_divisor_bits(&args), 127);
    }

    #[test]
    fn the_narrow_path_tightens_the_budget_instead_of_failing_late() {
        let (mut nodes, chain) = mult_chain(5, Some(SupportedOp::Linear(PolyOp::Add)));
        let stats = fuse_with(&mut nodes, &[], 64, false);

        assert_eq!(stats.fused, 3);
        assert_eq!(stats.rejected_over_budget, 1);
        assert_eq!(stats.max_divisor_bits_used, 3 * SCALE);
        assert_eq!(divisor_bits(&nodes, chain[2]), Some(3 * SCALE));
        assert_eq!(divisor_bits(&nodes, chain[4]), Some(2 * SCALE));

        let (mut nodes, chain) = mult_chain(5, Some(SupportedOp::Linear(PolyOp::Add)));
        let stats = fuse_with(&mut nodes, &[], 64, true);
        assert_eq!(stats.fused, 4);
        assert_eq!(stats.rejected_over_budget, 0);
        assert_eq!(divisor_bits(&nodes, chain[4]), Some(5 * SCALE));
    }

    #[test]
    fn an_explicit_budget_below_the_narrow_ceiling_still_wins() {
        let (mut nodes, _) = mult_chain(3, Some(SupportedOp::Linear(PolyOp::Add)));
        let stats = fuse_with(&mut nodes, &[], 2 * SCALE, false);
        assert_eq!(stats.fused, 1);
        assert_eq!(stats.rejected_over_budget, 1);
    }

    #[test]
    fn the_allowlists_are_the_narrow_ones_documented_above() {
        assert!(is_pure_rescale_inner(&SupportedOp::Linear(PolyOp::Mult)));
        assert!(!is_pure_rescale_inner(&SupportedOp::Linear(PolyOp::Add)));
        assert!(!is_pure_rescale_inner(&SupportedOp::Linear(
            PolyOp::Einsum {
                equation: "ij,jk->ik".to_string(),
            }
        )));

        assert!(can_absorb_delayed_scale(&SupportedOp::Linear(PolyOp::Mult)));
        assert!(can_absorb_delayed_scale(&SupportedOp::Linear(
            PolyOp::Einsum {
                equation: "ij,jk->ik".to_string(),
            }
        )));
        assert!(!can_absorb_delayed_scale(&SupportedOp::Linear(PolyOp::Add)));
        assert!(!can_absorb_delayed_scale(&SupportedOp::Linear(PolyOp::Sub)));
        assert!(!can_absorb_delayed_scale(&SupportedOp::Nonlinear(
            LookupOp::Sigmoid {
                scale: crate::circuit::utils::F32(128.0),
            }
        )));
    }
}
