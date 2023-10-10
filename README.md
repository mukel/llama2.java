# A Java port of Andrej Karpathy's llama2.c

This is a pure Java port of Andrej Karpathy's awesome [llama2.c](https://github.com/karpathy/llama2.c), a very simple implementation
to run inference of models with a [Llama2](https://arxiv.org/pdf/2302.13971.pdf)-like transformer-based LLM architecture.  

<p align="center">
  <img width="600" src="https://github.com/mukel/llama2.java/assets/1896283/66a8a650-f1a9-4540-9587-b112294e5e6b">
</p>

Currently, there isn't anything really original here, but I'll continue polishing it while keeping it in sync with the original.  
Besides the educational value, this project will be used to test and tune compiler optimizations on the JVM, particularly for the [Graal compiler](https://www.graalvm.org/latest/reference-manual/java/compiler).
This port used [llama2.scala](https://github.com/jrudolph/llama2.scala) initially as a reference.

## Build
Java 21+ is required, in particular the [`MemorySegment` mmap-ing feature](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/nio/channels/FileChannel.html#map(java.nio.channels.FileChannel.MapMode,long,long,java.lang.foreign.Arena)).  

The code expects [`tokenizer.bin`](https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin) in the current directory.
You can use [TinyStories](https://huggingface.co/karpathy/tinyllamas/tree/main) checkpoints or get LLama2 models by [following instructions](https://github.com/karpathy/llama2.c#metas-llama-2-models).

```bash
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

To build and run manually:
```bash
javac --enable-preview -source 21 --add-modules=jdk.incubator.vector Llama2.java
java --enable-preview --add-modules=jdk.incubator.vector Llama2 stories15M.bin
```

Or run it directly with [JBang](https://www.jbang.dev/):
```bash
jbang Llama2.java stories15M.bin
# With additional -D options and custom Java home.
JAVA_HOME=/path/to/java/home jbang -Djava.util.concurrent.ForkJoinPool.common.parallelism=0 -Dllama2.VectorAPI=false Llama2.java stories15M.bin
```

A `Makefile` and a `run.sh` script are also provided:

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
NATIVE_IMAGE_OPTIONS="--pgo-instrument -march=native --initialize-at-build-time=Llama2 -Dllama2.VectorAPI=false" \
make native-image

# Profile run to generate default.iprof, with no parallelism to speedup profiling.
./llama2 -Djava.util.concurrent.ForkJoinPool.common.parallelism=0 stories15M.bin

# Build optimized image
JAVA_HOME=$GRAALVM_HOME \
NATIVE_IMAGE_OPTIONS="--pgo -march=native --initialize-at-build-time=Llama2 -Dllama2.VectorAPI=false" \
make native-image

# Should run ~2X faster than regular image.
./llama2 stories15M.bin
```

## Performance

Quick numbers on an AMD Ryzen 3950X 64GB, Arch Linux.  
`llama2.java` executed on OpenJDK 20.0.2+9.  
To make things fair w.r.t. to vectorization, the Java version has a matmul implementation using the [Vector API](https://openjdk.org/jeps/448).  
In these measurements the JVM is warmed up enough to reach peak tokens/s.  
On GraalVM, please note that the Graal compiler doesn't support the Vector API yet, to avoid unexpected performance degradation, run with `-Dllama2.VectorAPI=false`.

****Notes**  
*The numbers below were collected using aggressive (gcc) compiler flags e.g. regular `gcc -O2 ...` wouldn't be as fast.*

### Single-threaded

`llama2.c` compiled with `gcc -Ofast -march=native run.c -lm -o run -march=native`  
`llama2.java` executed with `-Djava.util.concurrent.ForkJoinPool.common.parallelism=0`

| Model | Tokens per second | Speedup vs. llama2.c | Implementation |  
| ------|------------------ | -------------------- | -------------- | 
| stories15M.bin  |   363 |  1.0 | llama2.c    |
| stories15M.bin  |   237 | 0.65 | llama2.java |
| stories110M.bin | 51.71 |  1.0 | llama2.c    |
| stories110M.bin | 42.20 | 0.81 | llama2.java |
| llama2_7B.bin   |  0.92 |  1.0 | llama2.c    |
| llama2_7B.bin   |  0.88 | 0.95 | llama2.java |

### Multi-threaded

`llama2.c` compiled with `gcc -Ofast -fopenmp -march=native run.c -lm -o run -march=native`  
`llama2.c` executed with `OMP_NUM_THREADS=8`  
`llama2.java` executed with `-Djava.util.concurrent.ForkJoinPool.common.parallelism=8`  

| Model | Tokens per second | Speedup vs. llama2.c | Implementation |  
| ------|------------------ | -------------------- | -------------- |
|  stories15M.bin |  1233 |  1.0 | llama2.c    |
|  stories15M.bin |   438 | 0.35 | llama2.java |
| stories110M.bin |    90 |  1.0 | llama2.c    |
| stories110M.bin |    80 | 0.88 | llama2.java |
|   llama2_7B.bin |  1.68 |  1.0 | llama2.c    |
|   llama2_7B.bin |  1.65 | 0.98 | llama2.java |

****Notes**  
*In `stories15M.bin`, the C version shows a huge speedup, very likely a cache effect, this is considered an outlier.
Running with 16/32 threads may actually cause a slowdown; the performance is, in most cases, U-shaped w.r.t to the # of threads.
With that many threads, vectorization does not give any advantage, since throughput is limited by memory bandwidth.*

Performance is already comparable to the original C code, bar vectorization, even if the Java code has not been optimized yet.

## License

MIT
