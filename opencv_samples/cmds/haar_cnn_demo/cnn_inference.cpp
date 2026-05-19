// cnn_inference.cpp
#include <stdint.h>
#include <cmath>
#include "cnn_inference.h"

// Include quantized weights
#include "conv1_weight.h"
#include "conv1_bias.h"
#include "conv2_weight.h"
#include "conv2_bias.h"
#include "fc1_weight.h"
#include "fc1_bias.h"
#include "fc2_weight.h"
#include "fc2_bias.h"

// Layer structures
typedef struct {
    const int8_t* weight;
    float weight_scale;
    const int8_t* bias;
    float bias_scale;
    int in_channels;
    int out_channels;
    int kernel_size;
} ConvLayer;

typedef struct {
    const int8_t* weight;
    float weight_scale;
    const int8_t* bias;
    float bias_scale;
    int in_features;
    int out_features;
} FCLayer;

// Define layers with weights
static ConvLayer conv1 = {
    .weight = conv1_weight,
    .weight_scale = conv1_weight_scale,
    .bias = conv1_bias,
    .bias_scale = conv1_bias_scale,
    .in_channels = 1,
    .out_channels = 8,
    .kernel_size = 3
};

static ConvLayer conv2 = {
    .weight = conv2_weight,
    .weight_scale = conv2_weight_scale,
    .bias = conv2_bias,
    .bias_scale = conv2_bias_scale,
    .in_channels = 8,
    .out_channels = 16,
    .kernel_size = 3
};

static FCLayer fc1 = {
    .weight = fc1_weight,
    .weight_scale = fc1_weight_scale,
    .bias = fc1_bias,
    .bias_scale = fc1_bias_scale,
    .in_features = 16 * 8 * 8,
    .out_features = 32
};

static FCLayer fc2 = {
    .weight = fc2_weight,
    .weight_scale = fc2_weight_scale,
    .bias = fc2_bias,
    .bias_scale = fc2_bias_scale,
    .in_features = 32,
    .out_features = 1
};

// Helper functions
static void relu(float* x, int size) {
    for (int i = 0; i < size; i++) if (x[i] < 0) x[i] = 0;
}

static void maxpool2d(float* input, float* output, int h, int w, int c) {
    int out_h = h/2, out_w = w/2;
    for (int ch = 0; ch < c; ch++)
        for (int i = 0; i < out_h; i++)
            for (int j = 0; j < out_w; j++) {
                float max_val = -1e9;
                for (int pi = 0; pi < 2; pi++)
                    for (int pj = 0; pj < 2; pj++)
                        max_val = fmax(max_val, input[ch*h*w + (i*2+pi)*w + (j*2+pj)]);
                output[ch*out_h*out_w + i*out_w + j] = max_val;
            }
}

static void conv2d(float* input, float* output, ConvLayer* layer, int h, int w) {
    int out_h = h, out_w = w, k = layer->kernel_size, pad = 1;
    for (int oc = 0; oc < layer->out_channels; oc++)
        for (int oh = 0; oh < out_h; oh++)
            for (int ow = 0; ow < out_w; ow++) {
                float sum = 0;
                for (int ic = 0; ic < layer->in_channels; ic++)
                    for (int kh = 0; kh < k; kh++)
                        for (int kw = 0; kw < k; kw++) {
                            int ih = oh - pad + kh, iw = ow - pad + kw;
                            if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                int w_idx = oc * layer->in_channels * k * k + ic * k * k + kh * k + kw;
                                int in_idx = ic * h * w + ih * w + iw;
                                sum += input[in_idx] * layer->weight[w_idx] * layer->weight_scale;
                            }
                        }
                output[oc * out_h * out_w + oh * out_w + ow] = sum + layer->bias[oc] * layer->bias_scale;
            }
}

static void fc(float* input, float* output, FCLayer* layer) {
    for (int o = 0; o < layer->out_features; o++) {
        float sum = 0;
        for (int i = 0; i < layer->in_features; i++)
            sum += input[i] * layer->weight[o * layer->in_features + i] * layer->weight_scale;
        output[o] = sum + layer->bias[o] * layer->bias_scale;
    }
}

// Main inference function
float cnn_inference(uint8_t* image, int width, int height) {
    if (width != 34 || height != 34) return 0.5f;

    // Static local buffers (.bss SRAM)
    static float input[34 * 34];
    static float buffer1[8 * 34 * 34];
    static float buffer2[8 * 17 * 17];
    static float fc1_out[32];
    static float fc2_out[1];

    // Normalize input
    for (int i = 0; i < 34*34; i++) input[i] = image[i] / 255.0f;

    // Forward pass
    conv2d(input, buffer1, &conv1, 34, 34);
    relu(buffer1, 8 * 34 * 34);

    maxpool2d(buffer1, buffer2, 34, 34, 8);

    conv2d(buffer2, buffer1, &conv2, 17, 17);
    relu(buffer1, 16 * 17 * 17);

    maxpool2d(buffer1, buffer2, 17, 17, 16);

    // buffer2 holds flattened 16*8*8 tensor

    fc(buffer2, fc1_out, &fc1);
    relu(fc1_out, 32);

    fc(fc1_out, fc2_out, &fc2);

    // Sigmoid
    return 1.0f / (1.0f + expf(-fc2_out[0]));
}
