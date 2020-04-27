#include "feistel.h"

static unsigned char const k2[16] = {
    4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1};
static unsigned char const k1[16] = {
    13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7};

crypto_feistel_ctx *create_feistel_ctx(uint8_t it, uint8_t shift)
{
    crypto_feistel_ctx *ctx = malloc(sizeof(crypto_feistel_ctx));
    ctx->it = it;
    ctx->shift = shift;
    ctx->num = 8;
    //free(ctx->keys);
    ctx->keys = malloc(sizeof(feistel_subkeys) * ctx->it);
}

void delete_feistel_ctx(crypto_feistel_ctx *ctx)
{
    free(ctx->keys);
    free(ctx);
    //ctx = NULL;
}

/* Keys schedule from model page*/
void schedule(crypto_feistel_ctx *ctx, const uint8_t *key, unsigned int key_len)
{
    for (size_t i = 0; i < ctx->it; i++)
    {
        ctx->keys[i].key = key[0];
        //memcpy(ctx->keys[i].key, key, min(key_len, ctx->num / 8));
    }
}

int feistel_setkey(crypto_feistel_ctx *ctx, const uint8_t *key, unsigned int key_len)
{
    schedule(ctx, key, key_len);
    return 1;
}

inline static uint8_t f(uint8_t x)
{
    uint8_t res;
    res = k2[x >> 4 & 15] << 4 | k1[x & 15];

    return x;
}

inline static uint8_t shift(uint8_t x, uint8_t sh)
{
    return x << sh | x >> (8 - sh);
}

void feistel_generate(crypto_feistel_ctx *ctx, uint8_t *out, const uint8_t *in)
{
    uint8_t x2 = in[1];
    uint8_t x1 = in[0];
    uint8_t y1 = 0;
    uint8_t y2 = 0;

    for (uint8_t index = 0; index < ctx->it; index++)
    {
        uint8_t var = x2 + ctx->keys[index].key;
        var = f(var);
        var = shift(var, ctx->shift);
        y1 = x2;
        y2 = var ^ x1;

        /* Upgrade */
        x1 = y1;
        x2 = y2;
    }
    out[0] = y1;
    out[1] = y2;
}