// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract ChainLinkVerifier {
    uint256 private constant POINT = 64;

    function _point(bytes calldata proof, uint256 idx) private pure returns (bytes32 x, bytes32 y) {
        uint256 off = idx * POINT;
        require(off + POINT <= proof.length, "commitment index out of range");
        assembly {
            x := calldataload(add(proof.offset, off))
            y := calldataload(add(proof.offset, add(off, 32)))
        }
    }

    function checkLinks(bytes[] calldata proofs, uint256[] calldata outIdx, uint256[] calldata inIdx)
        public
        pure
        returns (bool)
    {
        require(outIdx.length == inIdx.length, "index arrays must match");
        for (uint256 t = 0; t + 1 < proofs.length; ++t) {
            for (uint256 j = 0; j < outIdx.length; ++j) {
                (bytes32 ax, bytes32 ay) = _point(proofs[t], outIdx[j]);
                (bytes32 bx, bytes32 by) = _point(proofs[t + 1], inIdx[j]);
                if (ax != bx || ay != by) return false;
            }
        }
        return true;
    }

    function checkAnchor(bytes calldata proof0, uint256[] calldata inIdx, bytes32[] calldata expected)
        public
        pure
        returns (bool)
    {
        require(expected.length == inIdx.length * 2, "expected must be x,y per index");
        for (uint256 j = 0; j < inIdx.length; ++j) {
            (bytes32 x, bytes32 y) = _point(proof0, inIdx[j]);
            if (x != expected[2 * j] || y != expected[2 * j + 1]) return false;
        }
        return true;
    }

    function checkPublicCommitments(bytes[] calldata proofs, uint256[] calldata idx, bytes32[] calldata expected)
        public
        pure
        returns (bool)
    {
        require(expected.length == proofs.length * idx.length * 2, "expected length mismatch");
        uint256 k = 0;
        for (uint256 t = 0; t < proofs.length; ++t) {
            for (uint256 j = 0; j < idx.length; ++j) {
                (bytes32 x, bytes32 y) = _point(proofs[t], idx[j]);
                if (x != expected[k] || y != expected[k + 1]) return false;
                k += 2;
            }
        }
        return true;
    }

    function verifyChain(
        address verifier,
        bytes[] calldata proofs,
        uint256[][] calldata instances,
        bytes32[] calldata vkaDigest,
        uint256[] calldata outIdx,
        uint256[] calldata inIdx
    ) external view returns (bool) {
        if (!checkLinks(proofs, outIdx, inIdx)) return false;
        for (uint256 t = 0; t < proofs.length; ++t) {
            (bool ok, bytes memory ret) = verifier.staticcall(
                abi.encodeWithSignature("verifyProof(bytes,uint256[],bytes32[])", proofs[t], instances[t], vkaDigest)
            );
            require(ok, "verifier call failed (possibly out of gas)");
            (bool verified,,) = abi.decode(ret, (bool, bytes32, int256[]));
            if (!verified) return false;
        }
        return true;
    }

    function noopBaseline(bytes[] calldata proofs, uint256[] calldata outIdx, uint256[] calldata inIdx)
        public
        pure
        returns (bool)
    {
        return proofs.length + outIdx.length + inIdx.length > 0;
    }
}
