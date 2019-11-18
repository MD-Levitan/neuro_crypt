#include <stdio.h>
#include <stdint.h>

typedef struct magma_subkeys magma_subkeys;
typedef struct crypto_tfm crypto_tfm;


/**
 * @brief Get from crypto instance magma keys
 * 
 * @param tfm		crypto instance
 * @return 			magma_subkeys 
 */
magma_subkeys *crypto_tfm_ctx(struct crypto_tfm *tfm);

/**
 * @brief Create a tfm ctx object
 * 
 * @return 			struct crypto instance
 */
struct crypto_tfm *create_tfm_ctx();

/**
 * @brief Destroy crypto instance 
 * 
 * @param tfm		crypto instance
 */
void delete_tfm_ctx(struct crypto_tfm *tfm);

/**
 * @brief Set key to crypto instance
 * 
 * @param tfm		crypto instance
 * @param key		ptr to key
 * @param key_len	key length (support only 32)
 * @return int		retun 1 if any error
 */
extern int magma_setkey(struct crypto_tfm *tfm, const uint8_t *key, 
						unsigned int key_len);

/**
 * @brief Encrypt block of data
 * 
 * @param tfm		crypto instance
 * @param out		result of encryption
 * @param in		plaintext block
 */
extern void magma_encrypt(struct crypto_tfm *tfm, uint8_t *out, const uint8_t *in);

/**
 * @brief Encyption of block by only i iteration
 * 
 * @param tfm		crypto instance
 * @param out		result of encryption
 * @param in		plaintext block
 * @param iter		iteration of GOST alg
 */
extern void magma_it(struct crypto_tfm *tfm, uint8_t *out,
					 const uint8_t *in, uint8_t iter);

/**
 * @brief Encyption of block by first n iterations
 * 
 * @param tfm		crypto instance
 * @param out		result of encryption
 * @param in		plaintext block
 * @param n			number of iteration
 */
extern void magma_it_n(struct crypto_tfm *tfm, uint8_t *out,
					   const uint8_t *in, uint8_t n);

/**
 * @brief Decrypt block of data
 * 
 * @param tfm		crypto instance
 * @param out		result of decryption
 * @param in		ciphertext block
 */
extern void magma_decrypt(struct crypto_tfm *tfm, uint8_t *out,
			  		 const uint8_t *in);					 

#define magma_step(tfm, out, in) \
	magma_it(tfm, out, in, 0)

