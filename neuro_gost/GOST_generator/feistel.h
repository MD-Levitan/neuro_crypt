#ifndef FEISTEL_H_
#define FEISTEL_H_

#include <stdio.h>
#include <stdint.h>

typedef struct feistel_subkeys feistel_subkeys;
typedef struct crypto_feistel_ctx crypto_feistel_ctx;

struct feistel_subkeys
{
    uint8_t key;
};

struct crypto_feistel_ctx
{
    /* Round Keys */
    feistel_subkeys *keys;

    /* N - number of bits, constant value */
    uint8_t num;

    /* Number of itearations */
    uint8_t it;

    /* Shift in algorithm */
    uint8_t shift;
};

/**
 * @brief Create a crypto_feistel context
 * 
 * @param it        number of iterations
 * @param shift     shift value
 * 
 * @return 			struct crypto instance
 */
crypto_feistel_ctx *create_feistel_ctx(uint8_t it, uint8_t shift);

/**
 * @brief Destroy crypto instance 
 * 
 * @param tfm		crypto instance
 */
void delete_feistel_ctx(crypto_feistel_ctx *ctx);

/**
 * @brief Set key to crypto instance
 * 
 * @param ctx		crypto instance
 * @param key		ptr to key
 * @param key_len	key length (support only 1)
 * @return int		retun 1 if any error
 */
int feistel_setkey(crypto_feistel_ctx *ctx, const uint8_t *key, unsigned int key_len);

/**
 * @brief Fucntion for creation pair for Neural Network.
 * 
 * @param ctx		crypto instance
 * @param out		ciphertext
 * @param in		plaintext
 */
void feistel_generate(crypto_feistel_ctx *ctx, uint8_t *out, const uint8_t *in);

#endif //FEISTEL_H_