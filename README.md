# A Java port of Andrej Karpathy's llama2.c

This is a Java port of Andrej Karpathy's awesome [llama2.c](https://github.com/karpathy/llama2.c), a very simple implementation
to run inference of models with a [Llama](https://arxiv.org/pdf/2302.13971.pdf)-like transformer-based LLM architecture.

The code expects [`tokenizer.bin`](https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin) in the current directory.

<p align="center">
  <img width="600" src="https://github.com/mukel/llama2.java/assets/1896283/f3e1ea49-b88f-4893-9656-e09057dc4281">  
</p>

Currently, there isn't anything original here. I'll continue improving it (e.g. use the Java Vector API) while
keeping it in sync with the original. This is a baseline implementation to gather preliminary performance data on the JVM.
This port is used [llama2.scala](https://github.com/jrudolph/llama2.scala) as a reference.

## Performance

Comparison on AMD Ryzen 3950X, Linux:

| Implementation | Tokens per second | Speedup |
| -------------- | ----------------- | ------- |
| llama2.c single-thread | 92 | 1.0x |
| llama2.c multi-thread (`gcc -O3 -fopenmp -lm ...`) | ~520 | 5.6x |
| llama2.java on GraalVM 23.0 (17) single-thread | 87 | 0.95x |

Performance is very close to the original C code, even if the Java code has not been optimized at all yet.

## License

MIT
