// Copyright (c) 2026 jesus luque.
//
// WHERE A GAUSSIAN'S TRANSFER KEEPS WHAT, told by how many values it is.
//
// A transfer is one run of values a gaussian (f16 on the device, floats in a
// record): the direct half, then the indirect one where it was kept, then
// the reflected field where a TX transfer kept it (docs/decisions.md, task
// TX). There is no mode token -- the count is the layout, as the presence of
// the data is the mode everywhere else in this tree:
//
//   9     degree 2, direct                   (the first transfer)
//   36    degree 2, direct and indirect      (the first transfer)
//   84    degree 2, both, and the field      (a TX transfer at --transfer-degree 2)
//   16    degree 3, direct
//   64    degree 3, direct and indirect
//   112   degree 3, both, and the field      (a TX transfer)
//
// The field is 16 rgb coefficients whatever the degree of the rest, and is
// only ever kept beside the indirect half, which is what couples it to a sky.
module transfer_layout;

/// The reflected field's values: sixteen rgb coefficients.
public static const uint kTransferFieldValues = 48;

/// Coefficients of the direct half: 9 or 16.
public uint transferDirectCount(uint count) {
    return (count == 16u || count == 64u || count == 112u) ? 16u : 9u;
}

/// Values of the indirect half: three times the direct one's, or none.
public uint transferIndirectCount(uint count) {
    const uint direct = transferDirectCount(count);
    return count >= 4u * direct ? 3u * direct : 0u;
}

/// Whether the reflected field follows the two halves.
public bool transferHasField(uint count) {
    const uint direct = transferDirectCount(count);
    return count == direct + 3u * direct + kTransferFieldValues;
}

/// Where the field starts, counted from the gaussian's first value.
public uint transferFieldOffset(uint count) {
    return transferDirectCount(count) + transferIndirectCount(count);
}
