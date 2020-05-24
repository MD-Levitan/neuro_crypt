#include "crypto.h"

crypto_tfm *create_crypto_tfm(ciphersuite_t suite, crypto_params *params)
{
    crypto_tfm *ctx = malloc(sizeof(crypto_tfm));
    crypto_magma_ctx *magma_ctx;
    crypto_feistel_ctx *feistel_ctx;

    switch (suite)
    {
    case MAGMA:
        magma_ctx = create_magma_ctx();
        ctx->magma = magma_ctx;
        if (params)
        {
            ctx->magma->input = params->magma_params.input;
        }
        break;

    case FEISTEL:
        feistel_ctx = create_feistel_ctx(params ? params->feistel_params.iter : 1,
                                         params ? params->feistel_params.shift : 1);
        ctx->feistel = feistel_ctx;
        break;
    default:
        free(ctx);
        ctx = NULL;
        break;
    }
    return ctx;
}

void delete_crypto_tfm(ciphersuite_t suite, crypto_tfm *ctx)
{
    switch (suite)
    {
    case MAGMA:
        if (ctx && ctx->magma)
        {
            delete_magma_ctx(ctx->magma);
        }
        break;
    case FEISTEL:
        if (ctx && ctx->feistel)
        {
            delete_feistel_ctx(ctx->feistel);
        }
        break;
    default:
        break;
    }
    free(ctx);
    ctx = NULL;
}

int setkey_crypto_tfm(ciphersuite_t suite, crypto_tfm *ctx, const uint8_t *key, unsigned int key_len)
{
    switch (suite)
    {
    case MAGMA:
        if (ctx && ctx->magma)
        {
            return magma_setkey(ctx->magma, key, key_len);
        }
    case FEISTEL:
        if (ctx && ctx->feistel)
        {
            return feistel_setkey(ctx->feistel, key, key_len);
        }
    default:
        return 1;
    }
}
