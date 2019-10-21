#include <stdio.h>
#include <stdint.h>

typedef struct magma_subkeys magma_subkeys;
typedef struct crypto_tfm crypto_tfm;


magma_subkeys *crypto_tfm_ctx(struct crypto_tfm *tfm);
struct crypto_tfm *create_tfm_ctx();
void delete_tfm_ctx(struct crypto_tfm *tfm);

extern int magma_setkey(struct crypto_tfm *tfm, const uint8_t *key, 
						unsigned int key_len);
extern void magma_encrypt(struct crypto_tfm *tfm, uint8_t *out, const uint8_t *in);
extern void magma_it(struct crypto_tfm *tfm, uint8_t *out,
					 const uint8_t *in, uint8_t iter);
extern void magma_decrypt(struct crypto_tfm *tfm, uint8_t *out,
			  		 const uint8_t *in);					 

#define magma_step(tfm, out, in) \
	magma_it(tfm, out, in, 0)

