#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "magma.h"

static uint8_t key[32] = //{ 0x01 };
{
 	0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78, 0x89,
 	0x9A, 0xAB, 0xBC, 0xCD, 0xDE, 0xEF, 0xFF, 0x01,
 	0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78, 0x89,
 	0x9A, 0xAB, 0xBC, 0xCD, 0xDE, 0xEF, 0xFF, 0x01,
};

const char *usage = "Usage:  ./generator <SIZE> <GEN_TYPE> <MODEL>";

typedef void (*generator)(struct crypto_tfm *ctx, uint64_t size,
						  char *filename_x, char *filename_y,
						  void (*generator)(struct crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in));

typedef void (*generator_model)(struct crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

#define gen_n(n) 

/* Consecutive generator */
void iterate_generator(struct crypto_tfm *ctx, uint64_t size, 
					   char *filename_x, char *filename_y, 
					   void (*generator)(struct crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in))
{
	FILE *out_file_x, *out_file_y;
	uint64_t in;

	out_file_x = fopen(filename_x, "wb");
	out_file_y = fopen(filename_y, "wb");	

	if (out_file_x == NULL || out_file_y == NULL) 
	{
		printf("error: cannot open file!\n");
		exit(1);
	}
	
	for (in = 0; in < size; ++in)
	{
		generator(ctx, out_file_x, out_file_y, in);	
	}

	fclose(out_file_x);
	fclose(out_file_y);	
}

/* Consecutive generator */
void iterate_parallel_generator(struct crypto_tfm *ctx, uint64_t size, 
					   char *filename_x, char *filename_y, 
					   void (*generator)(struct crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in))
{
	
	FILE *out_file_x, *out_file_y;
	uint64_t in;
	uint32_t n1, n2;

	out_file_x = fopen(filename_x, "wb");
	out_file_y = fopen(filename_y, "wb");	

	if (out_file_x == NULL || out_file_y == NULL) 
	{
		printf("error: cannot open file!\n");
		exit(1);
	}
	
	for (n1 = 0; n1 < size; ++n1)
	{
		for (n2 = 0; n2 < size; ++n2)
		{
			in = 0;
			((uint32_t *) &in)[0] = SWAP_32(n1);
			((uint32_t *) &in)[1] = SWAP_32(n2);
			generator(ctx, out_file_x, out_file_y, in);	
		}
	}

	fclose(out_file_x);
	fclose(out_file_y);	
}

/* Random generator */
void random_generator(struct crypto_tfm *ctx, uint64_t size, 
					  char *filename_x, char *filename_y, 
					  void (*generator)(struct crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in))
{
	FILE *out_file_x, *out_file_y;
	uint64_t in;

	out_file_x = fopen(filename_x, "wb");
	out_file_y = fopen(filename_y, "wb");	

	if (out_file_x == NULL || out_file_y == NULL) 
	{
		printf("error: cannot open file!\n");
		exit(1);
	}

	srand(time(NULL));
	
	for (uint64_t i = 0; i < size; ++i)
	{
		in = 0;
		((uint32_t *) &in)[0] = rand();
		((uint32_t *) &in)[1] = rand();
		generator(ctx, out_file_x, out_file_y, in);	
	}

	fclose(out_file_x);
	fclose(out_file_y);	
}


/* Consecutive generator with random order */
void random_iterate_generator(struct crypto_tfm *ctx, uint64_t size, 
					  		  char *filename_x, char *filename_y, 
					  		  void (*generator)(struct crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in))
{
	FILE *out_file_x, *out_file_y;
	uint64_t in, value, size_x, *memory;

	out_file_x = fopen(filename_x, "wb");
	out_file_y = fopen(filename_y, "wb");	

	if (out_file_x == NULL || out_file_y == NULL) 
	{
		printf("error: cannot open file!\n");
		exit(1);
	}

	srand(time(NULL));
	memory = malloc(sizeof(uint64_t) * size);
	if (memory == NULL)
	{
		printf("error: cannot malloc!\n");
		exit(1);
	}

	for (uint64_t i = 0; i < size; i++)
	{
		memory[i] = i;
	}
	
	size_x = size;
	for (uint64_t i = 0; i < size - 1; ++i)
	{
		in = 0;
		value = 0;
		((uint32_t *) &in)[0] = rand();
		((uint32_t *) &in)[1] = rand();
		value = memory[in % size_x];
		//printf("%ld\n", value);
		memory[in % size_x] = memory[size_x - 1];
		size_x--;
		
		generator(ctx, out_file_x, out_file_y, value);	
	}

	fclose(out_file_x);
	fclose(out_file_y);	
}

// /* Consecutive generator with random order*/
// void random_iterate_parallel_generator(struct crypto_tfm *ctx, uint64_t size, 
// 					  		  		   char *filename_x, char *filename_y, 
// 					  		           void (*generator)(struct crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in))
// {
// 	FILE *out_file_x, *out_file_y;
// 	uint64_t in, value, size_x, *memory1, *memory2;

// 	out_file_x = fopen(filename_x, "wb");
// 	out_file_y = fopen(filename_y, "wb");	

// 	if (out_file_x == NULL || out_file_y == NULL) 
// 	{
// 		printf("error: cannot open file!\n");
// 		exit(1);
// 	}

// 	srand(time(NULL));
// 	memory2 = malloc(sizeof(uint64_t) * size);
// 	memory1 = malloc(sizeof(uint64_t) * size);
// 	if (memory1 == NULL || memory2 == NULL)
// 	{
// 		printf("error: cannot malloc!\n");
// 		exit(1);
// 	}

// 	for (uint64_t i = 0; i < size; i++)
// 	{
// 		memory1[i] = i;
// 	}
	
// 	for (uint64_t i = 0; i < size; i++)
// 	{
// 		memory2[i] = i;
// 	}

// 	size_x = size;
// 	for (uint64_t i = 0; i < size - 1; ++i)
// 	{
// 		in = 0;
// 		value = 0;
// 		((uint32_t *) &in)[0] = rand();
// 		((uint32_t *) &in)[1] = rand();
// 		value = memory1[in % size_x];
// 		//printf("%ld\n", value);
// 		memory1[in % size_x] = memory1[size_x - 1];
// 		size_x--;
		
// 		generator(ctx, out_file_x, out_file_y, value);	
// 	}

// 	fclose(out_file_x);
// 	fclose(out_file_y);	
// }

/* Generator for G3 model/1-round of GOST */
void round_generator(struct crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in)
{
	uint64_t out;

	magma_it(ctx, (uint8_t *) &out, (uint8_t *) &in, 0);
	fwrite((uint8_t *) &in, sizeof(uint8_t), 8, out_file_x);
	fwrite((uint8_t *) &out, sizeof(uint8_t), 8, out_file_y);
}

/* Generator for N-round GOST */
void n_round_generator_(struct crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in, uint8_t n)
{
	uint64_t out;

	magma_it_n(ctx, (uint8_t *) &out, (uint8_t *) &in, n);
	fwrite((uint8_t *) &in, sizeof(uint8_t), 8, out_file_x);
	fwrite((uint8_t *) &out, sizeof(uint8_t), 8, out_file_y);
}

/* Generator for 2-round GOST */
void n_round_generator(struct crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in)
{
	uint64_t out;

	magma_it_n(ctx, (uint8_t *) &out, (uint8_t *) &in, 2);
	fwrite((uint8_t *) &in, sizeof(uint8_t), 8, out_file_x);
	fwrite((uint8_t *) &out, sizeof(uint8_t), 8, out_file_y);
}

/* Generator for G2 model */
void primitive_g2_generator_(struct crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in)
{
	
	uint64_t out;
	uint32_t x, y;
	magma_neuro_g2(ctx, (uint8_t *) &out, (uint8_t *) &in, &x, &y);
	uint8_t p = 0; 
	//for (uint8_t p = 0; p < 8; ++p)
	//{
		uint8_t var = ((uint8_t *) &in)[4 + p / 2];
		var = p % 2 ?  (var & 0xF) : (var >> 4 & 0xF);

		uint8_t var1 = ((uint8_t *) &x)[p / 2];
		var1 = p % 2 ?  (var1 & 0xF) : (var1 >> 4 & 0xF);
		var = (var << 4) | var1;

		uint8_t var2 = ((uint8_t *) &y)[p / 2];
		var2 = p % 2 ?  (var2 & 0xF) : (var2 >> 4 & 0xF);
			
		fwrite(&var, sizeof(uint8_t), 1, out_file_x);
		fwrite(&var2, sizeof(uint8_t), 1, out_file_y);
	//}	
}

/* Generator for G1 model */
void primitive_g1_generator_(struct crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in)
{
	uint64_t out;
	uint32_t x, y;

	magma_neuro_g1(ctx, (uint8_t *) &out, (uint8_t *) &in, &x, &y);

	// printf("%llx\n", x);
	// printf("%llx\n", y);
	// printf("%llx\n", in);
	// printf("%llx\n", out);

	uint8_t p = 0; 
	//for (uint8_t p = 0; p < 8; ++p)
	//{
		uint8_t var = ((uint8_t *) &in)[4 + p / 2];
		var = p % 2 ?  (var & 0xF) : (var >> 4 & 0xF);

		uint8_t var1 = ((uint8_t *) &x)[p / 2];
		var1 = p % 2 ?  (var1 & 0xF) : (var1 >> 4 & 0xF);
		var = (var << 4) | var1;

		uint8_t var2 = ((uint8_t *) &y)[p / 2];
		var2 = p % 2 ?  (var2 & 0xF) : (var2 >> 4 & 0xF);
			
		fwrite(&var, sizeof(uint8_t), 1, out_file_x);
		fwrite(&var2, sizeof(uint8_t), 1, out_file_y);
	//}
		
}

/* Generator for G0 model */
void primitive_g0_generator_(struct crypto_tfm *ctx, FILE * out_file_x, FILE * out_file_y, uint64_t in)
{
	uint64_t out;
	uint32_t x, y;
	
	magma_neuro_g0(ctx, (uint8_t *) &out, (uint8_t *) &in, &x, &y);

	// printf("%llx\n", x);
	// printf("%llx\n", y);
	// printf("%llx\n", in);
	// printf("%llx\n", out);

	uint8_t p = 0; 
	//for (uint8_t p = 0; p < 8; ++p)
	//{
		uint8_t var = ((uint8_t *) &in)[4 + p / 2];
		var = p % 2 ?  (var & 0xF) : (var >> 4 & 0xF);

		uint8_t var1 = ((uint8_t *) &x)[p / 2];
		var1 = p % 2 ?  (var1 & 0xF) : (var1 >> 4 & 0xF);
		var = (var << 4) | var1;

		uint8_t var2 = ((uint8_t *) &y)[p / 2];
		var2 = p % 2 ?  (var2 & 0xF) : (var2 >> 4 & 0xF);
			
		fwrite(&var, sizeof(uint8_t), 1, out_file_x);
		fwrite(&var2, sizeof(uint8_t), 1, out_file_y);
	//}	
}

/* Generator for G0 model */
void primitive_g0_generator(struct crypto_tfm *ctx, FILE * out_file_x, FILE * out_file_y, uint64_t in)
{
	uint64_t out;
	uint32_t y;
	uint32_t n1 = GETU32_BE(((uint8_t *) &in));
	uint32_t n2 = GETU32_BE(((uint8_t *) &in) + 4);

	
	magma_neuro_g0_primitive(ctx, n1, n2, &y);

	uint8_t var = ((uint8_t *) &n1)[0];		
	var = var & 0xF;

	uint8_t var1 = ((uint8_t *) &n2)[0];
	var1 = var1 & 0xF;
	var = (var << 4) | var1;

	uint8_t var2 = ((uint8_t *) &y)[0];
	var2 = var2 & 0xF;
			
	fwrite(&var, sizeof(uint8_t), 1, out_file_x);
	fwrite(&var2, sizeof(uint8_t), 1, out_file_y);	
}

/* Generator for G0 model */
void primitive_g1_generator(struct crypto_tfm *ctx, FILE * out_file_x, FILE * out_file_y, uint64_t in)
{
	uint64_t out;
	uint32_t y;
	uint32_t n1 = GETU32_BE(((uint8_t *) &in));
	uint32_t n2 = GETU32_BE(((uint8_t *) &in) + 4);

	
	magma_neuro_g1_primitive(ctx, n1, n2, &y);

	uint8_t var = ((uint8_t *) &n1)[0];		
	var = var & 0xF;

	uint8_t var1 = ((uint8_t *) &n2)[0];
	var1 = var1 & 0xF;
	var = (var << 4) | var1;

	uint8_t var2 = ((uint8_t *) &y)[0];
	var2 = var2 & 0xF;
			
	fwrite(&var, sizeof(uint8_t), 1, out_file_x);
	fwrite(&var2, sizeof(uint8_t), 1, out_file_y);	
}

/* Generator for G2 model */
void primitive_g2_generator(struct crypto_tfm *ctx, FILE * out_file_x, FILE * out_file_y, uint64_t in)
{
	uint64_t out;
	uint32_t y;
	uint32_t n1 = GETU32_BE(((uint8_t *) &in));
	uint32_t n2 = GETU32_BE(((uint8_t *) &in) + 4);

	magma_neuro_g2_primitive(ctx, n1, n2, &y);

	uint8_t var = ((uint8_t *) &n1)[0];		
	var = var & 0xF;

	uint8_t var1 = ((uint8_t *) &n2)[0];
	var1 = var1 & 0xF;
	var = (var << 4) | var1;

	uint8_t var2 = ((uint8_t *) &y)[0];
	var2 = var2 & 0xF;
			
	fwrite(&var, sizeof(uint8_t), 1, out_file_x);
	fwrite(&var2, sizeof(uint8_t), 1, out_file_y);	
}

/* Generator for G3 model */
void primitive_g3_generator(struct crypto_tfm *ctx, FILE * out_file_x, FILE * out_file_y, uint64_t in)
{
	uint64_t out;
	uint32_t y;
	uint32_t n1 = GETU32_BE(((uint8_t *) &in));
	uint32_t n2 = GETU32_BE(((uint8_t *) &in) + 4);

	magma_neuro_g3_primitive(ctx, n1, n2, &y);
	out = SWAP_32(y);


	fwrite((uint8_t *) &in, sizeof(uint8_t), 8, out_file_x);
	fwrite((uint8_t *) &out, sizeof(uint8_t), 8, out_file_y);
}

/*
 *	Generator
 *  Usage:  ./generator <SIZE> <GEN_TYPE> <MODEL>
 *
 */
int main(int argc, const char **argv) {
	if (argc < 4) 
	{
		printf("%s!\n", usage);
		return -1;
	}
	
	int8_t gen_v, mod_v;
	uint64_t size = 0;
	
	struct crypto_tfm *ctx;
	
	size = strtoll(argv[1], NULL, 10);
	gen_v = strtol(argv[2], NULL, 10);
	mod_v = strtol(argv[3], NULL, 10);
	
	ctx = create_tfm_ctx();
	magma_setkey(ctx, key, sizeof(key));

	generator generator_ptr;
	generator_model generator_model_ptr;
	char *in, *out;

	switch(gen_v)
	{ 
		case 0: 
		{ 
			generator_ptr = iterate_generator;
			break;
		}
		case 1: 
		{
			generator_ptr = random_generator;
			break;
		}
		case 2: 
		{
			generator_ptr = random_iterate_generator;
			break;
		}
		case 3: 
		{
			generator_ptr = iterate_parallel_generator;
			break;
		}
		default:
		{
			generator_ptr = random_generator;
			break;
		}
	}

	switch(mod_v)
	{ 
		case 0: 
		{ 
			generator_model_ptr = primitive_g0_generator;
			in = "bin/g0_x.bin";
			out = "bin/g0_y.bin";
			break;
		}
		case 1: 
		{ 
			generator_model_ptr = primitive_g1_generator;
			in = "bin/g1_x.bin";
			out = "bin/g1_y.bin";
			break;
		}
		case 2: 
		{ 
			generator_model_ptr = primitive_g2_generator;
			in = "bin/g2_x.bin";
			out = "bin/g2_y.bin";
			break;
		}
		case 3: 
		{ 
			generator_model_ptr = primitive_g3_generator;
			in = "bin/g3_x.bin";
			out = "bin/g3_y.bin";
			break;
		}
		case 4: 
		{ 
			generator_model_ptr = round_generator;
			in = "bin/n1_x.bin";
			out = "bin/n1_y.bin";
			break;
		}
		case 5: 
		{ 
			generator_model_ptr = n_round_generator;
			in = "bin/n2_x.bin";
			out = "bin/n2_y.bin";
			break;
		}
		default:
		{
			generator_model_ptr = round_generator;
			in = "bin/n1_x.bin";
			out = "bin/n1_y.bin";
			break;
		}
	}
	generator_ptr(ctx, size, in, out, generator_model_ptr);
	

	delete_tfm_ctx(ctx);
}

// // test
// int main()
// {
// 	struct crypto_tfm *ctx;
// 	ctx = create_tfm_ctx();
// 	magma_setkey(ctx, key, sizeof(key));

// 	uint64_t in, out1, out2, y, x;
// 	((uint32_t *) &in)[0] = 11111111;
// 	((uint32_t *) &in)[1] = 22222222;

// 	magma_neuro(ctx, &out2, &in, &y, &x);
// 	_magma_it(ctx, &out1, &in, 0);

// 	printf("First: %llx\n", out1);
// 	printf("Second: %llx\n", out2);
// }
