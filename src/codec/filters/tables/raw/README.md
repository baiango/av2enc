# Verbatim AV2 v1.0.0 attachments

Do not edit these files.  They are the normative table headers referenced by
§9.8 and §9.9 of the AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, spec commit `38f28e7`).

Source (both were compared byte-for-byte):

- `https://av2.aomedia.org/v1.0.0/attachments/<name>`
- `https://raw.githubusercontent.com/AOMediaCodec/av2-spec/42d091fe9f8ea4b526800d073b3e4c4da132967e/v1.0.0/attachments/<name>`

Retrieved 2026-11-11.  `SHA256SUMS` pins the bytes; the generator refuses to
run if a file does not match.

To refresh (only if a new spec revision is adopted):

```sh
base=https://av2.aomedia.org/v1.0.0/attachments
for f in gdf_alpha.h gdf_weight.h gdf_bias.h gdf_intra_error.h \
         gdf_inter_error.h pc_wiener_filters.h pc_wiener_lut_to_class.h \
         pc_wiener_sub_classify.h pc_wiener_sub_classify2.h gaussian_sequence.h; do
  curl -fsS "$base/$f" -o "$f"
done
shasum -a 256 *.h > SHA256SUMS
```

Then update the expected hashes in `../generate_spec_tables.py`, regenerate,
and review the generated-header diff before committing.
