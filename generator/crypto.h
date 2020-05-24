#ifndef CRYPTO_H_
#define CRYPTO_H_

#include "magma.h"
#include "feistel.h"

typedef union crypto_tfm crypto_tfm;
typedef union crypto_params crypto_params;
typedef enum ciphersuite_t ciphersuite_t;

enum ciphersuite_t
{
    MAGMA,
    FEISTEL
};

union crypto_tfm {
    crypto_magma_ctx *magma;
    crypto_feistel_ctx *feistel;
};

union crypto_params {
    struct
    {
        uint8_t shift;
        uint8_t iter;
    } feistel_params;
    struct
    {
        uint8_t input;
    } magma_params;
};

/**
 * @brief Create a crypto_context
 * 
 * @param suite     cipher suite
 * @param it        number of iterations
 * @param shift     shift value
 * 
 * @return 			struct crypto instance
 */
crypto_tfm *create_crypto_tfm(ciphersuite_t suite, crypto_params *params);

/**
 * @brief Destroy crypto instance 
 * 
 * @param suite     cipher suite
 * @param tfm		crypto instance
 */
void delete_crypto_tfm(ciphersuite_t suite, crypto_tfm *ctx);

/**
 * @brief Set key to crypto instance
 * 
 * @param suite     cipher suite
 * @param ctx		crypto instance
 * @param key		ptr to key
 * @param key_len	key length
 * @return int		retun 1 if any error
 */
int setkey_crypto_tfm(ciphersuite_t suite, crypto_tfm *ctx, const uint8_t *key, unsigned int key_len);

#endif //CRYPTO_H_