# A Java port of Andrej Karpathy's llama2.c

This is a pure Java port of Andrej Karpathy's awesome [llama2.c](https://github.com/karpathy/llama2.c), a very simple implementation
to run inference of models with a [Llama2](https://arxiv.org/pdf/2302.13971.pdf)-like transformer-based LLM architecture.  

<p align="center">
  <img width="600" src="https://github.com/mukel/llama2.java/assets/1896283/c7db4110-1bf6-466c-9fac-130a6ecefe8a">
</p>

Currently, there isn't anything really original here, but I'll continue polishing it while keeping it in sync with the original.  
Besides the educational value, this project will be used to test and tune compiler optimizations on the JVM, particularly for the [Graal compiler](https://www.graalvm.org/latest/reference-manual/java/compiler).
This port used [llama2.scala](https://github.com/jrudolph/llama2.scala) initially as a reference.

## Build
Java 20+ is required, in particular the [`MemorySegment` mmap-ing feature](https://docs.oracle.com/en/java/javase/20/docs/api/java.base/java/nio/channels/FileChannel.html#map(java.nio.channels.FileChannel.MapMode,long,long,java.lang.foreign.SegmentScope)).  
The code expects [`tokenizer.bin`](https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin) in the current directory.

To build and run manually:
```bash
javac --enable-preview -source 20 Llama2.java
java --enable-preview Llama2 stories15M.bin
```

For convenience, a `Makefile` and a `run.sh` script are also provided:

```bash
make # optional, run.sh already runs make

JAVA_HOME=$GRAALVM_HOME \
JAVA_RUNTIME_OPTIONS=-Djava.util.concurrent.ForkJoinPool.common.parallelism=8 \
./run.sh stories15M.bin
```

#### Native image

A standalone native image can be created with [GraalVM](https://www.graalvm.org/)
```bash
JAVA_HOME=$GRAALVM_HOME NATIVE_IMAGE_OPTIONS="-march=native" make native-image
./llama2 stories15M.bin
```

Or can also be built with [Profile-Guided Optimizations (PGO)](https://www.graalvm.org/dev/reference-manual/native-image/guides/optimize-native-executable-with-pgo), on Oracle GaaalVM:
```bash
JAVA_HOME=$GRAALVM_HOME \
NATIVE_IMAGE_OPTIONS="--pgo-instrument -march=native" \
make native-image

# Profile run to generate default.iprof, with no parallelism to speedup profiling.
./llama2 -Djava.util.concurrent.ForkJoinPool.common.parallelism=0 stories15M.bin

# Build optimized image
JAVA_HOME=$GRAALVM_HOME \
NATIVE_IMAGE_OPTIONS="--pgo -march=native" \
make native-image

./llama2 stories15M.bin
```

## Performance

Quick numbers on an AMD Ryzen 3950X 64GB, Linux:
`llama2.java` executed on Oracle GraalVM 20 (23.0.1)
All programs are executed with [nucleus sampling](https://arxiv.org/pdf/1904.09751.pdf) disabled (e.g. `-p 0`), since the Java version implements a faster algorithm.
This difference barely affects larger models e.g. >= `stories110M.bin` but is very noticeable in smaller models.  

****Notes**  
*The numbers below were collected using aggressive (gcc) compiler flags e.g. regular `gcc -O2 ...` wouldn't be as fast.*

### Single-threaded

`llama2.c` compiled with `gcc -Ofast -march=native run.c -lm -o run -march=native`  
`llama2.java` executed with `-Djava.util.concurrent.ForkJoinPool.common.parallelism=0`

| Implementation | Model | Tokens per second | Speedup vs. llama2.c |
| ---------------| ------|------------------ | -------------------- |
| llama2.c    | stories15M.bin  |   363 |  1.0 |
| llama2.java | stories15M.bin  |    86 | 0.23 |
| llama2.c    | stories110M.bin | 52.76 |  1.0 |
| llama2.java | stories110M.bin | 12.89 | 0.24 |
| llama2.c    | llama2_7B.bin   |  0.93 |  1.0 |
| llama2.java | llama2_7B.bin   |  0.21 | 0.23 |

****Notes**  
*The **~4X** difference, consistent across benchmarks, is due to the vectorization of matmul, which the Java implementation lacks.
Surprinsingly, neither C2 nor Graal auto-vectorization optimizations are applied in matmul.*

### Multi-threaded

`llama2.c` compiled with `gcc -Ofast -fopenmp -march=native run.c -lm -o run -march=native`

| Implementation | Model | Tokens per second | Speedup vs. llama2.c |
| ---------------| ------|------------------ | -------------------- |
| llama2.c    |  stories15M.bin |  1233 |  1.0 |
| llama2.java |  stories15M.bin |   309 | 0.25 |
| llama2.c    | stories110M.bin | 56.60 |  1.0 |
| llama2.java | stories110M.bin | 65.85 | **1.16** |
| llama2.c    |   llama2_7B.bin |  1.64 |  1.0 |
| llama2.java |   llama2_7B.bin |  1.56 | 0.95 |

****Notes**  
*In `stories15M.bin`, the C version shows a huge speedup, very likely a cache effect, this is considered an outlier.
Running with 16/32 threads may actually cause a slowdown; the performance is, in most cases, U-shaped w.r.t to the # of threads.
With that many threads, vectorization does not give any advantage, since throughput is limited by memory bandwidth.*

Performance is already comparable to the original C code, bar vectorization, even if the Java code has not been optimized yet.

## License

MIT
