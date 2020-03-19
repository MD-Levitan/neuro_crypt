#include <stdio.h>
#include <stdint.h>

typedef struct magma_subkeys magma_subkeys;
typedef struct crypto_tfm crypto_tfm;

#define GETU32_BE(pt) (\
	((uint32_t)(pt)[0] << 24) | \
	((uint32_t)(pt)[1] << 16) | \
	((uint32_t)(pt)[2] <<  8) | \
	((uint32_t)(pt)[3]) )

#define GETU32_LE(pt) (\
	((uint32_t)(pt)[3] << 24) | \
	((uint32_t)(pt)[2] << 16) | \
	((uint32_t)(pt)[1] <<  8) | \
	((uint32_t)(pt)[0]) )

#define SWAP_32(pt) (\
	((uint32_t)((uint8_t*) &pt)[0] << 24) | \
	((uint32_t)((uint8_t*) &pt)[1] << 16) | \
	((uint32_t)((uint8_t*) &pt)[2] <<  8) | \
	((uint32_t)((uint8_t*) &pt)[3] ))

/* This function checks whether CPU supports Intel's instruction BSWAP.
   If so, this instruction will be used, portable code otherwise. */
#ifndef BSWAP32
#define BSWAP32(n) __builtin_bswap32(n)
#endif

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

extern void _magma_encrypt(struct crypto_tfm *tfm, uint8_t *out, const uint8_t *in);

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

extern void _magma_it(struct crypto_tfm *tfm, uint8_t *out,
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

/**
 * @brief Fucntion for creation pair for Neural Network.
 * You can find description of this function(model) in report as model g_0.
 * 
 * @param tfm		crypto instance
 * @param out		result of encryption
 * @param y			y for neural network
 */
extern void magma_neuro_g0(struct crypto_tfm *tfm, uint8_t *out,
						   const uint8_t *in, uint32_t *x, uint32_t *y);

/**
 * @brief Fucntion for creation pair for Neural Network.
 * You can find description of this function(model) in report as model g_1.
 * 
 * @param tfm		crypto instance
 * @param out		result of encryption
 * @param y			y for neural network
 */
extern void magma_neuro_g1(struct crypto_tfm *tfm, uint8_t *out,
						   const uint8_t *in, uint32_t *x, uint32_t *y);

/**
 * @brief Fucntion for creation pair for Neural Network.
 * You can find description of this function(model) in report as model g_2.
 * 
 * @param tfm		crypto instance
 * @param out		result of encryption
 * @param y			y for neural network
 */
extern void magma_neuro_g2(struct crypto_tfm *tfm, uint8_t *out,
					 const uint8_t *in, uint32_t *x, uint32_t *y);



extern void magma_neuro_g0_primitive(struct crypto_tfm *tfm, uint32_t n1, uint32_t n2, uint32_t *y);

extern void magma_neuro_g1_primitive(struct crypto_tfm *tfm, uint32_t n1, uint32_t n2, uint32_t *y);

extern void magma_neuro_g2_primitive(struct crypto_tfm *tfm, uint32_t n1, uint32_t n2, uint32_t *y);

extern void magma_neuro_g3_primitive(struct crypto_tfm *tfm, uint32_t n1, uint32_t n2, uint32_t *y);

extern void magma_neuro_g4_primitive(struct crypto_tfm *tfm, uint32_t n1, uint32_t n2, uint32_t *y);


#define magma_step(tfm, out, in) \
	magma_it(tfm, out, in, 0)

