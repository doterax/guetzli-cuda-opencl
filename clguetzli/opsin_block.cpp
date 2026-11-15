/*
 * Host (CPU) copies of the kernel helpers that host code needs.
 *
 * These used to be reached by compiling clguetzli.cl itself as C++
 * (clguetzli.cl.cpp). The bodies below are verbatim ports of the kernel
 * functions and must stay numerically identical to them: they feed
 * the GPU comparator's per-block opsin lists for the GPU block search.
 */
#include "opsin_block.h"

#include <cmath>

namespace guetzli {
namespace {

float kSrgb8ToLinearTableF[256] = {
    0.000000f,   0.077399f,   0.154799f,   0.232198f,   0.309598f,   0.386997f,   0.464396f,   0.541796f,   0.619195f,
    0.696594f,   0.773994f,   0.853367f,   0.937509f,   1.026303f,   1.119818f,   1.218123f,   1.321287f,   1.429375f,
    1.542452f,   1.660583f,   1.783830f,   1.912253f,   2.045914f,   2.184872f,   2.329185f,   2.478910f,   2.634105f,
    2.794824f,   2.961123f,   3.133055f,   3.310673f,   3.494031f,   3.683180f,   3.878171f,   4.079055f,   4.285881f,
    4.498698f,   4.717556f,   4.942502f,   5.173584f,   5.410848f,   5.654341f,   5.904108f,   6.160196f,   6.422649f,
    6.691512f,   6.966827f,   7.248640f,   7.536993f,   7.831928f,   8.133488f,   8.441715f,   8.756651f,   9.078335f,
    9.406810f,   9.742115f,   10.084290f,  10.433375f,  10.789410f,  11.152432f,  11.522482f,  11.899597f,  12.283815f,
    12.675174f,  13.073712f,  13.479465f,  13.892470f,  14.312765f,  14.740385f,  15.175366f,  15.617744f,  16.067555f,
    16.524833f,  16.989614f,  17.461933f,  17.941824f,  18.429322f,  18.924460f,  19.427272f,  19.937793f,  20.456054f,
    20.982090f,  21.515934f,  22.057618f,  22.607175f,  23.164636f,  23.730036f,  24.303404f,  24.884774f,  25.474176f,
    26.071642f,  26.677203f,  27.290891f,  27.912736f,  28.542769f,  29.181020f,  29.827520f,  30.482299f,  31.145387f,
    31.816813f,  32.496609f,  33.184802f,  33.881422f,  34.586499f,  35.300062f,  36.022139f,  36.752760f,  37.491953f,
    38.239746f,  38.996169f,  39.761248f,  40.535013f,  41.317491f,  42.108710f,  42.908697f,  43.717481f,  44.535088f,
    45.361546f,  46.196882f,  47.041124f,  47.894297f,  48.756429f,  49.627547f,  50.507676f,  51.396845f,  52.295078f,
    53.202402f,  54.118843f,  55.044428f,  55.979181f,  56.923129f,  57.876298f,  58.838712f,  59.810398f,  60.791381f,
    61.781686f,  62.781338f,  63.790363f,  64.808784f,  65.836627f,  66.873918f,  67.920679f,  68.976937f,  70.042715f,
    71.118037f,  72.202929f,  73.297414f,  74.401516f,  75.515259f,  76.638668f,  77.771765f,  78.914575f,  80.067122f,
    81.229428f,  82.401518f,  83.583415f,  84.775142f,  85.976722f,  87.188178f,  88.409534f,  89.640813f,  90.882037f,
    92.133229f,  93.394412f,  94.665609f,  95.946841f,  97.238133f,  98.539506f,  99.850982f,  101.172584f, 102.504334f,
    103.846254f, 105.198366f, 106.560693f, 107.933256f, 109.316077f, 110.709177f, 112.112579f, 113.526305f, 114.950375f,
    116.384811f, 117.829635f, 119.284868f, 120.750532f, 122.226647f, 123.713235f, 125.210317f, 126.717914f, 128.236047f,
    129.764737f, 131.304005f, 132.853871f, 134.414357f, 135.985483f, 137.567270f, 139.159738f, 140.762907f, 142.376799f,
    144.001434f, 145.636832f, 147.283012f, 148.939997f, 150.607804f, 152.286456f, 153.975971f, 155.676371f, 157.387673f,
    159.109900f, 160.843070f, 162.587203f, 164.342319f, 166.108438f, 167.885578f, 169.673761f, 171.473005f, 173.283330f,
    175.104755f, 176.937299f, 178.780982f, 180.635824f, 182.501843f, 184.379058f, 186.267489f, 188.167154f, 190.078073f,
    192.000265f, 193.933749f, 195.878543f, 197.834666f, 199.802137f, 201.780975f, 203.771198f, 205.772826f, 207.785876f,
    209.810367f, 211.846319f, 213.893748f, 215.952674f, 218.023115f, 220.105089f, 222.198615f, 224.303711f, 226.420395f,
    228.548685f, 230.688599f, 232.840156f, 235.003373f, 237.178269f, 239.364861f, 241.563167f, 243.773205f, 245.994993f,
    248.228549f, 250.473890f, 252.731035f, 255.000000f,
};

// Deliberately not const (matches the old C++ build of the kernel, where
// __constant expanded to nothing): with -ffast-math, constant coefficients get
// folded and reassociated, which changes results in the last bits.
float g_mix[12] = {
    0.348036746003f,  0.577814843137f, 0.0544556093735f, 0.774145581713f, 0.26922717275f,  0.767247733938f,
    0.0366922708552f, 0.920130265014f, 0.0882062883536f, 0.158581714673f, 0.712857943858f, 10.6524069248f,
};

float g_gamma_p[5 + 1] = {
    881.979476556478289f, 1496.058452015812463f, 908.662212739659481f,
    373.566100223287378f, 85.840860336314364f,   6.683258861509244f,
};

float g_gamma_q[5 + 1] = {
    12.262350348616792f, 20.557285797683576f, 12.161463238367844f,
    4.711532733641639f,  0.899112889751053f,  0.035662329617191f,
};

int _min_i(int a, int b) {
  return a < b ? a : b;
}

void OpsinAbsorbance(const float in[3], float out[3]) {
  out[0] = g_mix[0] * in[0] + g_mix[1] * in[1] + g_mix[2] * in[2] + g_mix[3];
  out[1] = g_mix[4] * in[0] + g_mix[5] * in[1] + g_mix[6] * in[2] + g_mix[7];
  out[2] = g_mix[8] * in[0] + g_mix[9] * in[1] + g_mix[10] * in[2] + g_mix[11];
}

float EvaluatePolynomial(const float x, const float* coefficients, int n) {
  float b1 = 0.0f;
  float b2 = 0.0f;

  for (int i = n - 1; i >= 0; i--) {
    if (i == 0) {
      const float x_b1 = x * b1;
      b1 = x_b1 - b2 + coefficients[0];
      break;
    }
    const float x_b1 = x * b1;
    const float t = (x_b1 + x_b1) - b2 + coefficients[i];
    b2 = b1;
    b1 = t;
  }

  return b1;
}

float Gamma(float v) {
  const float min_value = 0.770000000000000f;
  const float max_value = 274.579999999999984f;
  const float x01 = (v - min_value) / (max_value - min_value);
  const float xc = 2.0f * x01 - 1.0f;

  const float yp = EvaluatePolynomial(xc, g_gamma_p, 6);
  const float yq = EvaluatePolynomial(xc, g_gamma_q, 6);
  if (yq == 0.0f) return 0.0f;
  return (float)(yp / yq);
}

void RgbToXyb(const float r, const float g, const float b, float* valx, float* valy, float* valz) {
  const float a0 = 1.01611726948f;
  const float a1 = 0.982482243696f;
  const float a2 = 1.43571362627f;
  const float a3 = 0.896039849412f;
  *valx = a0 * r - a1 * g;
  *valy = a2 * r + a3 * g;
  *valz = b;
}

void Convolution(const unsigned int xsize, const unsigned int ysize, const int xstep, const int len, const int offset,
                 const float* multipliers, const float* inp, const float border_ratio, float* result) {
  float weight_no_border = 0;

  for (int j = 0; j <= 2 * offset; ++j) {
    weight_no_border += multipliers[j];
  }
  for (int x = 0, ox = 0; x < (int)xsize; x += xstep, ox++) {
    int minx = x < offset ? 0 : x - offset;
    int maxx = _min_i((int)xsize, (int)(x + len - offset)) - 1;
    float weight = 0.0f;
    for (int j = minx; j <= maxx; ++j) {
      weight += multipliers[j - x + offset];
    }
    // Interpolate linearly between the no-border scaling and border scaling.
    weight = (1.0f - border_ratio) * weight + border_ratio * weight_no_border;
    float scale = 1.0f / weight;
    for (unsigned int y = 0; y < ysize; ++y) {
      float sum = 0.0f;
      for (int j = minx; j <= maxx; ++j) {
        sum += inp[y * xsize + j] * multipliers[j - x + offset];
      }
      result[ox * ysize + y] = (float)(sum * scale);
    }
  }
}

void BlurEx(const float* r, const int xsize, const int ysize, const float border_ratio, float* output) {
  // sigma = 1.1 (the only value the kernel uses).
  const float scaler = -0.41322314049586772f;
  const int diff = 2;
  const int expn_size = 5;
  float expn[5] = {std::exp(scaler * (-diff) * (-diff)), std::exp(scaler * (-diff + 1) * (-diff + 1)),
                   std::exp(scaler * (-diff + 2) * (-diff + 2)), std::exp(scaler * (-diff + 3) * (-diff + 3)),
                   std::exp(scaler * (-diff + 4) * (-diff + 4))};
  const int xstep = 1;
  const int ystep = xstep;

  int dxsize = (xsize + xstep - 1) / xstep;

  float tmp[8 * 8];  // fully written by the first pass
  Convolution(xsize, ysize, xstep, expn_size, diff, expn, r, border_ratio, tmp);
  Convolution(ysize, dxsize, ystep, expn_size, diff, expn, tmp, border_ratio, output);
}

void OpsinDynamicsImageBlock(float* r, float* g, float* b, const float* r_blurred, const float* g_blurred,
                             const float* b_blurred, const int size) {
  for (int i = 0; i < size; ++i) {
    float sensitivity[3];
    {
      // Calculate sensitivity[3] based on the smoothed image gamma derivative.
      float pre_rgb[3] = {r_blurred[i], g_blurred[i], b_blurred[i]};
      float pre_mixed[3];
      OpsinAbsorbance(pre_rgb, pre_mixed);
      sensitivity[0] = Gamma(pre_mixed[0]) / pre_mixed[0];
      sensitivity[1] = Gamma(pre_mixed[1]) / pre_mixed[1];
      sensitivity[2] = Gamma(pre_mixed[2]) / pre_mixed[2];
    }
    float cur_rgb[3] = {r[i], g[i], b[i]};
    float cur_mixed[3];
    OpsinAbsorbance(cur_rgb, cur_mixed);
    cur_mixed[0] *= sensitivity[0];
    cur_mixed[1] *= sensitivity[1];
    cur_mixed[2] *= sensitivity[2];
    float x, y, z;
    RgbToXyb(cur_mixed[0], cur_mixed[1], cur_mixed[2], &x, &y, &z);
    r[i] = (float)(x);
    g[i] = (float)(y);
    b[i] = (float)(z);
  }
}

}  // namespace

const float* Srgb8ToLinearTableF() {
  return kSrgb8ToLinearTableF;
}

void OpsinDynamicsBlock8x8(float rgb[3][kOpsinBlockSize]) {
  float rgb_blurred[3][kOpsinBlockSize];
  for (int i = 0; i < 3; ++i) {
    BlurEx(rgb[i], 8, 8, 0, rgb_blurred[i]);
  }
  OpsinDynamicsImageBlock(rgb[0], rgb[1], rgb[2], rgb_blurred[0], rgb_blurred[1], rgb_blurred[2], kOpsinBlockSize);
}

}  // namespace guetzli
