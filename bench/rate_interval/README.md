# rate-interval coder microbenchmark

Compares the R19 interval-width accumulator with the R05-style trial encoder on
the same §8.3 symbol sequence (one fresh sink per 512-symbol leaf, adaptation
on). Build and run:

```sh
c++ -std=c++20 -O2 -ffp-contract=off -I src \
    bench/rate_interval/bench_rate_interval.cpp \
    src/codec/transform/tx_partition/tx_partition.cpp -o /tmp/ribench
/tmp/ribench
```

The emitted `bits/leaf` for the two sinks should differ by under a bit; the ns
numbers are only meaningful relative to each other on the same host.
