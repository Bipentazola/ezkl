// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract T2OnChainGas {
    function receiveLogits(uint256[] calldata logits) external pure returns (uint256) {
        return logits.length == 0 ? 0 : logits[logits.length - 1];
    }

    function argmaxOf(uint256[] calldata logits) external pure returns (uint256 idx) {
        uint256 best;
        for (uint256 i = 0; i < logits.length; ++i) {
            if (logits[i] > best) {
                best = logits[i];
                idx = i;
            }
        }
    }

    function msmWithGeneratorBase(uint256[] calldata scalars) external view returns (uint256 ax, uint256 ay) {
        uint256 n = scalars.length;
        if (n == 0) return (0, 0);
        (ax, ay) = _mul(1, 2, scalars[0]);
        for (uint256 i = 1; i < n; ++i) {
            (uint256 px, uint256 py) = _mul(1, 2, scalars[i]);
            (ax, ay) = _add(ax, ay, px, py);
        }
    }

    function t2Step(uint256[] calldata logits, uint256 expectedX, uint256 expectedY)
        external
        view
        returns (bool commitmentOk, uint256 idx)
    {
        uint256 n = logits.length;
        uint256 ax;
        uint256 ay;
        if (n > 0) {
            (ax, ay) = _mul(1, 2, logits[0]);
        }
        uint256 best;
        for (uint256 i = 1; i < n; ++i) {
            (uint256 px, uint256 py) = _mul(1, 2, logits[i]);
            (ax, ay) = _add(ax, ay, px, py);
        }
        for (uint256 i = 0; i < n; ++i) {
            if (logits[i] > best) {
                best = logits[i];
                idx = i;
            }
        }
        commitmentOk = (ax == expectedX && ay == expectedY);
    }

    function _mul(uint256 x, uint256 y, uint256 s) private view returns (uint256 rx, uint256 ry) {
        assembly {
            let p := mload(0x40)
            mstore(p, x)
            mstore(add(p, 0x20), y)
            mstore(add(p, 0x40), s)
            if iszero(staticcall(gas(), 0x07, p, 0x60, p, 0x40)) { revert(0, 0) }
            rx := mload(p)
            ry := mload(add(p, 0x20))
        }
    }

    function _add(uint256 x1, uint256 y1, uint256 x2, uint256 y2) private view returns (uint256 rx, uint256 ry) {
        assembly {
            let p := mload(0x40)
            mstore(p, x1)
            mstore(add(p, 0x20), y1)
            mstore(add(p, 0x40), x2)
            mstore(add(p, 0x60), y2)
            if iszero(staticcall(gas(), 0x06, p, 0x80, p, 0x40)) { revert(0, 0) }
            rx := mload(p)
            ry := mload(add(p, 0x20))
        }
    }
}
