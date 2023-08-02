/*
Inference for Llama-2 Transformer model in pure Java.

Example compile: (see README for more details)
$ javac Llama2.java

Then run with:
$ java Llama2
*/
// ----------------------------------------------------------------------------
// Transformer and RunState structs, and related memory management

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

final class Config {
    final int dim; // transformer dimension
    final int hidden_dim; // for ffn layers
    final int n_layers; // number of layers
    final int n_heads; // number of query heads
    final int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    final int vocab_size; // vocabulary size, usually 256 (byte-level)
    final int seq_len; // max sequence length
    final boolean shared_weights;
    final int head_size;

    Config(ByteBuffer buffer) {
        this.dim = buffer.getInt();
        this.hidden_dim = buffer.getInt();
        this.n_layers = buffer.getInt();
        this.n_heads = buffer.getInt();
        this.n_kv_heads = buffer.getInt();
        int vocab_size = buffer.getInt();
        this.vocab_size = Math.abs(vocab_size);
        this.seq_len = buffer.getInt();
        this.shared_weights = vocab_size > 0;
        this.head_size = dim / n_heads;
    }
}

final class Weights {
    // token embedding table
    final float[] token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    final float[] rms_att_weight; // (layer, dim) rmsnorm weights
    // weights for matmuls
    final float[] wq; // (layer, dim, dim)
    final float[] wk; // (layer, dim, dim)
    final float[] wv; // (layer, dim, dim)
    final float[] wo; // (layer, dim, dim)
    final float[] rms_ffn_weight; // (layer, dim)
    // weights for ffn
    final float[] w1; // (layer, hidden_dim, dim)
    final float[] w2; // (layer, dim, hidden_dim)
    final float[] w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    final float[] rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    final float[] freq_cis_real; // (seq_len, dim/2)
    final float[] freq_cis_imag; // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    final float[] wcls;

    static float[] take(FloatBuffer buffer, int... dims) {
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        float[] floats = new float[n];
        buffer.get(floats);
        return floats;
    }

    Weights(Config config, FloatBuffer buffer) {
        this.token_embedding_table = take(buffer, config.vocab_size, config.dim);
        this.rms_att_weight = take(buffer, config.n_layers, config.dim);
        this.wq = take(buffer, config.n_layers, config.dim, config.dim);
        this.wk = take(buffer, config.n_layers, config.dim, config.dim);
        this.wv = take(buffer, config.n_layers, config.dim, config.dim);
        this.wo = take(buffer, config.n_layers, config.dim, config.dim);
        this.rms_ffn_weight = take(buffer, config.n_layers, config.dim);
        this.w1 = take(buffer, config.n_layers, config.hidden_dim, config.dim);
        this.w2 = take(buffer, config.n_layers, config.dim, config.hidden_dim);
        this.w3 = take(buffer, config.n_layers, config.hidden_dim, config.dim);
        this.rms_final_weight = take(buffer, config.dim);
        this.freq_cis_real = take(buffer, config.seq_len, config.head_size / 2);
        this.freq_cis_imag = take(buffer, config.seq_len, config.head_size / 2);
        this.wcls = config.shared_weights ? this.token_embedding_table : null;
    }
}

final class RunState {
    // current wave of activations
    final float[] x; // activation at current time stamp (dim,)
    final float[] xb; // same, but inside a residual branch (dim,)
    final float[] xb2; // an additional buffer just for convenience (dim,)
    final float[] hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    final float[] hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    final float[] q; // query (dim,)
    final float[] k; // key (dim,)
    final float[] v; // value (dim,)
    final float[] att; // buffer for scores/attention values (n_heads, seq_len)
    final float[] logits; // output logits
    // kv cache
    final float[] key_cache;   // (layer, seq_len, dim)
    final float[] value_cache; // (layer, seq_len, dim)

    RunState(Config config) {
        this.x = new float[config.dim];
        this.xb = new float[config.dim];
        this.xb2 = new float[config.dim];
        this.hb = new float[config.hidden_dim];
        this.hb2 = new float[config.hidden_dim];
        this.q = new float[config.dim];
        this.k = new float[config.dim];
        this.v = new float[config.dim];
        this.att = new float[config.n_heads * config.seq_len];
        this.logits = new float[config.vocab_size];
        this.key_cache = new float[config.n_layers * config.seq_len * config.dim];
        this.value_cache = new float[config.n_layers * config.seq_len * config.dim];
    }
}

public class Llama2 {

// ----------------------------------------------------------------------------
// initialization: read from checkpoint

// ----------------------------------------------------------------------------
// neural net blocks

    static void accum(float[] a, float[] b, int size) {
        for (int i = 0; i < size; i++) {
            a[i] += b[i];
        }
    }

    static void rmsnorm(float[] o, float[] x, float[] weight, int weightOffset, int size) {
        // calculate sum of squares
        float ss = 0.0f;
        for (int j = 0; j < size; j++) {
            ss += x[j] * x[j];
        }
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / (float) Math.sqrt(ss);
        // normalize and scale
        for (int j = 0; j < size; j++) {
            o[j] = weight[weightOffset + j] * (ss * x[j]);
        }
    }

    static void softmax(float[] x, int xOffset, int size) {
        // find max value (for numerical stability)
        float max_val = x[0 + xOffset];
        for (int i = 1; i < size; i++) {
            if (x[i + xOffset] > max_val) {
                max_val = x[i + xOffset];
            }
        }
        // exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[i + xOffset] = (float) Math.exp(x[i + xOffset] - max_val);
            sum += x[i + xOffset];
        }
        // normalize
        for (int i = 0; i < size; i++) {
            x[i + xOffset] /= sum;
        }
    }

    static void matmul(float[] xout, float[] x, float[] w, int wOffset, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            for (int j = 0; j < n; j++) {
                val += w[wOffset + i * n + j] * x[j];
            }
            xout[i] = val;
        }
    }

    static void transformer(int token, int pos, Config p, RunState s, Weights w) {

        // a few convenience variables
        int dim = p.dim;
        int hidden_dim = p.hidden_dim;
        int head_size = p.head_size;

        // copy the token embedding into x
        System.arraycopy(w.token_embedding_table, token * dim, s.x, 0, dim);

        // forward all the layers
        for (int l = 0; l < p.n_layers; l++) {

            // attention rmsnorm
            rmsnorm(s.xb, s.x, w.rms_att_weight, dim * l, dim);

            // qkv matmuls for this position
            matmul(s.q, s.xb, w.wq, dim * dim * l, dim, dim);
            matmul(s.k, s.xb, w.wk, dim * dim * l, dim, dim);
            matmul(s.v, s.xb, w.wv, dim * dim * l, dim, dim);

            // apply RoPE rotation to the q and k vectors for each head
            for (int h = 0; h < p.n_heads; h++) {
                // get the q and k vectors for this head
                int qOffset = h * head_size;
                int kOffset = h * head_size;
                // float* q = s->q + h * head_size;
                // float* k = s->k + h * head_size;
                // rotate q and k by the freq_cis_real and freq_cis_imag
                for (int i = 0; i < head_size; i += 2) {
                    float q0 = s.q[qOffset + i];
                    float q1 = s.q[qOffset + i + 1];
                    float k0 = s.k[kOffset + i];
                    float k1 = s.k[kOffset + i + 1];
                    float fcr = w.freq_cis_real[pos * head_size / 2 + i / 2];
                    float fci = w.freq_cis_imag[pos * head_size / 2 + i / 2];
                    s.q[qOffset + i] = q0 * fcr - q1 * fci;
                    s.q[qOffset + i + 1] = q0 * fci + q1 * fcr;
                    s.k[kOffset + i] = k0 * fcr - k1 * fci;
                    s.k[kOffset + i + 1] = k0 * fci + k1 * fcr;
                }
            }

            // save key,value at this time step (pos) to our kv cache
            int loff = l * p.seq_len * dim; // kv cache layer offset for convenience

            System.arraycopy(s.k, 0, s.key_cache, loff + pos * dim, dim);
            System.arraycopy(s.v, 0, s.value_cache, loff + pos * dim, dim);

            // multihead attention. iterate over all heads
            for (int h = 0; h < p.n_heads; h++) {
                // get the query vector for this head
                // float* q = s.q + h * head_size;
                int qOffset = h * head_size;

                // attention scores for this head
                // float* att = s.att + h * p.seq_len;
                int attOffset = h * p.seq_len;

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= pos; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s.key_cache + loff + t * dim + h * head_size;
                    int keyCacheOffset = loff + t * dim + h * head_size;
                    // calculate the attention score as the dot product of q and k
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) {
                        score += s.q[qOffset + i] * s.key_cache[keyCacheOffset + i];
                    }
                    score /= (float) Math.sqrt(head_size);
                    // save the score to the attention buffer
                    s.att[attOffset + t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(s.att, attOffset, pos + 1);

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * head_size;
                int xbOffset = h * head_size;
                // memset(xb, 0, head_size * sizeof(float));

                for (int i = 0; i < head_size; ++i) {
                    s.xb[xbOffset + i] = 0f;
                }

                for (int t = 0; t <= pos; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s.value_cache + loff + t * dim + h * head_size;
                    int vOffset = loff + t * dim + h * head_size;
                    // get the attention weight for this timestep
                    float a = s.att[attOffset + t];
                    // accumulate the weighted value inconfigto xb
                    for (int i = 0; i < head_size; i++) {
                        s.xb[xbOffset + i] += a * s.value_cache[vOffset + i];
                    }
                }
            }

            // final matmul to get the output of the attention
            matmul(s.xb2, s.xb, w.wo, dim * dim * l, dim, dim);

            // residual connection back into x
            accum(s.x, s.xb2, dim);

            // ffn rmsnorm
            rmsnorm(s.xb, s.x, w.rms_ffn_weight, dim * l, dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            matmul(s.hb, s.xb, w.w1, dim * p.hidden_dim * l, dim, p.hidden_dim);
            matmul(s.hb2, s.xb, w.w3, p.hidden_dim * dim * l, dim, p.hidden_dim);

            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            for (int i = 0; i < hidden_dim; i++) {
                s.hb[i] = s.hb[i] / (1.0f + (float) Math.exp(-s.hb[i]));
            }

            // elementwise multiply with w3(x)
            for (int i = 0; i < hidden_dim; i++) {
                s.hb[i] = s.hb[i] * s.hb2[i];
            }

            // final matmul to get the output of the ffn
            // matmul(s.xb, s.hb, w.w2 + l*dim*hidden_dim, hidden_dim, dim);
            matmul(s.xb, s.hb, w.w2, dim * p.hidden_dim * l, p.hidden_dim, dim);

            // residual connection
            accum(s.x, s.xb, dim);
        }

        // final rmsnorm
        rmsnorm(s.x, s.x, w.rms_final_weight, 0, dim);

        // classifier into logits
        matmul(s.logits, s.x, w.wcls, 0, dim, p.vocab_size);
    }

// ----------------------------------------------------------------------------
// byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt

    static int str_lookup(String str, String[] vocab, int vocab_size) {
        // find the first perfect match for str in vocab, return its index or -1 if not found
        for (int i = 0; i < vocab_size; i++) {
            if (str.equals(vocab[i])) {
                return i;
            }
        }
        return -1;
    }

    static int bpe_encode(String text, String[] vocab, float[] vocab_scores, int vocab_size, int[] tokens) {
        // first encode every individual byte in the input string
        int n_tokens = 0; // the number of tokens
        for (int i = 0; i < text.length(); ++i) {
            char c = text.charAt(i);
            String singleChar = String.valueOf(c);
            int id = str_lookup(singleChar, vocab, vocab_size);
            if (id == -1) {
                System.out.printf("not good\n");
                System.exit(1);
            }
            tokens[n_tokens] = id;
            n_tokens++;
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;

            for (int i = 0; i < n_tokens - 1; ++i) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                String str_buffer = vocab[tokens[i]] + vocab[tokens[i + 1]];
                // sprintf(str_buffer, "%s%s", vocab[tokens[i]], vocab[tokens[i+1]]);
                int id = str_lookup(str_buffer, vocab, vocab_size);
                if (id != -1 && vocab_scores[id] > best_score) {
                    // this merge pair exists in vocab! record its score and position
                    best_score = vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1) {
                break; // we couldn't find any more pairs to merge, so we're done
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id;
            // delete token at position best_idx+1, shift the entire sequence back 1
            for (int i = best_idx + 1; i < n_tokens - 1; i++) {
                tokens[i] = tokens[i + 1];
            }
            n_tokens--; // token length decreased
        }

        return n_tokens;
    }

// ----------------------------------------------------------------------------
// utilities

    static long time_in_ms() {
        // return time in milliseconds, for benchmarking the model speed
        return System.nanoTime() / 1_000_000;
    }

    static long rng_seed;

    static int random_u32() {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        rng_seed ^= rng_seed >> 12;
        rng_seed ^= rng_seed << 25;
        rng_seed ^= rng_seed >> 27;
        return (int) ((rng_seed * 0x2545F4914F6CDD1DL) >> 32);
    }

    static float random_f32() { // random float32 in [0,1)
        return (random_u32() >>> 8) / 16777216.0f;
    }

    static int sample(float[] probabilities, int n) {
        // sample index from probabilities, they must sum to 1
        float r = random_f32();
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += probabilities[i];
            if (r < cdf) {
                return i;
            }
        }
        return n - 1; // in case of rounding errors
    }

    static int argmax(float[] v, int n) {
        // return argmax of v in elements 0..n
        int max_i = 0;
        float max_p = v[0];
        for (int i = 1; i < n; i++) {
            if (v[i] > max_p) {
                max_i = i;
                max_p = v[i];
            }
        }
        return max_i;
    }

// ----------------------------------------------------------------------------

    public static void main(String[] args) throws IOException {

        // poor man's C argparse
        String checkpoint = null; // e.g. out/model.bin
        float temperature = 0.0f; // 0.9f; // e.g. 1.0, or 0.0
        int steps = 256;          // max number of steps to run for, 0: use seq_len
        String prompt = null;     // prompt string

        // 'checkpoint' is necessary arg
        if (args.length < 1) {
            System.out.println("Usage: java Llama2 <checkpoint_file> [temperature] [steps] [prompt]\n");
            System.exit(1);
        }
        if (args.length >= 1) {
            checkpoint = args[0];
        }
        if (args.length >= 2) {
            // optional temperature. 0.0 = (deterministic) argmax sampling. 1.0 = baseline
            temperature = Float.parseFloat(args[1]);
        }
        if (args.length >= 3) {
            steps = Integer.parseInt(args[2]);
        }
        if (args.length >= 4) {
            prompt = args[3];
        }

        // seed rng with time. if you want deterministic behavior use temperature 0.0
        rng_seed = System.currentTimeMillis() / 1000; // (unsigned int)time(NULL);

        Config config;
        Weights weights;

        try (FileChannel fileChannel = FileChannel.open(Paths.get(checkpoint), StandardOpenOption.READ)) {
            ByteBuffer bb = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
            bb.order(ByteOrder.LITTLE_ENDIAN);
            // read in the config header
            config = new Config(bb);
            weights = new Weights(config, bb.asFloatBuffer());
        }

        // right now we cannot run for more than config.seq_len steps
        if (steps <= 0 || steps > config.seq_len) {
            steps = config.seq_len;
        }

        // read in the tokenizer.bin file
        String[] vocab = new String[config.vocab_size];
        float[] vocab_scores = new float[config.vocab_size];
        try (FileChannel channel = FileChannel.open(Paths.get("tokenizer.bin"), StandardOpenOption.READ)) {
            ByteBuffer bb = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            bb.order(ByteOrder.LITTLE_ENDIAN);
            int max_token_length = bb.getInt();
            for (int i = 0; i < config.vocab_size; i++) {
                vocab_scores[i] = bb.getFloat();
                int len = bb.getInt();
                byte[] bytes = new byte[len];
                bb.get(bytes);
                vocab[i] = new String(bytes);
            }
        }

        // create and init the application RunState
        RunState state = new RunState(config);

        // process the prompt, if any
        int[] prompt_tokens = null;
        int num_prompt_tokens = 0;
        if (prompt != null) {
            prompt_tokens = new int[config.seq_len];
            num_prompt_tokens = bpe_encode(prompt, vocab, vocab_scores, config.vocab_size, prompt_tokens);
        }

        // start the main loop
        long start = 0;  // used to time our code, only initialized after first iteration
        int next;        // will store the next token in the sequence
        int token = 1;   // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
        int pos = 0;     // position in the sequence
        System.out.println("<s>"); // explicit print the initial BOS token for stylistic symmetry
                                   // reasons
        while (pos < steps) {

            // forward the transformer to get logits for the next token
            transformer(token, pos, config, state, weights);

            if (pos < num_prompt_tokens) {
                // if we are still processing the input prompt, force the next prompt token
                next = prompt_tokens[pos];
            } else {
                // sample the next token
                if (temperature == 0.0f) {
                    // greedy argmax sampling: take the token with the highest probability
                    next = argmax(state.logits, config.vocab_size);
                } else {
                    // apply the temperature to the logits
                    for (int q = 0; q < config.vocab_size; q++) {
                        state.logits[q] /= temperature;
                    }
                    // apply softmax to the logits to get the probabilities for next token
                    softmax(state.logits, 0, config.vocab_size);
                    // we sample from this distribution to get the next token
                    next = sample(state.logits, config.vocab_size);
                }
            }

            // following BOS token (1), sentencepiece decoder strips any leading whitespace (see PR
            // #89)
            String token_str = (token == 1 && vocab[next].charAt(0) == ' ') ? vocab[next].substring(1) : vocab[next];
            System.out.printf("%s", token_str);
            System.out.flush();

            // advance forward
            token = next;
            pos++;
            // init our timer here because the first iteration is slow due to memmap
            if (start == 0) {
                start = time_in_ms();
            }
        }

        // report achieved tok/s
        long end = time_in_ms();
        System.out.printf("\nachieved tok/s: %f\n", (steps - 1) / (double) (end - start) * 1000);
    }
}
